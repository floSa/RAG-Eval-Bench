"""Acces aux modeles, derriere une API OpenAI-compatible.

C'est la decision d'architecture qui porte tout le projet : Ollama et vLLM
exposent tous deux `/v1`, et *tous* les frameworks d'evaluation (Ragas,
DeepEval, TruLens, RAGChecker, promptfoo) acceptent un `base_url` custom.
Donc un seul point d'entree suffit pour brancher n'importe quel framework
sur n'importe quel moteur, sans adaptateur maison et sans appel externe.

Deux regles de fonctionnement :

1. Les appels sont async avec un semaphore. Une campagne d'evaluation, c'est
   des milliers d'appels independants ; les enchainer en serie transforme
   20 minutes en 4 heures.
2. Chaque appel renvoie sa consommation de tokens et sa latence. Le cout est
   une metrique d'evaluation a part entiere, pas un detail d'exploitation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .settings import Settings, settings as default_settings


@dataclass
class Completion:
    text: str
    model: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Usage:
    """Compteur cumulatif, par role de modele."""

    calls: dict[str, int] = field(default_factory=dict)
    prompt_tokens: dict[str, int] = field(default_factory=dict)
    completion_tokens: dict[str, int] = field(default_factory=dict)

    def add(self, role: str, c: Completion) -> None:
        self.calls[role] = self.calls.get(role, 0) + 1
        self.prompt_tokens[role] = self.prompt_tokens.get(role, 0) + c.prompt_tokens
        self.completion_tokens[role] = (
            self.completion_tokens.get(role, 0) + c.completion_tokens
        )

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class LLMClient:
    """Client unique pour les 4 roles (generator / embedder / judge / verifier).

    Le role n'est pas cosmetique : il sert a ventiler la consommation de
    tokens, ce qui permet de repondre a « combien me coute mon evaluation
    par rapport a mon pipeline ? » — souvent un facteur 5 ou plus.
    """

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or default_settings
        self._client = AsyncOpenAI(
            base_url=self.cfg.llm_base_url,
            api_key=self.cfg.llm_api_key,
            timeout=self.cfg.llm_timeout_s,
            max_retries=0,  # gere par tenacity, pour tracer les echecs
        )
        self._sem = asyncio.Semaphore(self.cfg.llm_concurrency)
        self.usage = Usage()

    # -- generation --------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        role: str = "generator",
        temperature: float = 0.0,
        max_tokens: int = 512,
        system: str | None = None,
        seed: int | None = 42,
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        @retry(
            stop=stop_after_attempt(self.cfg.llm_max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        async def _call() -> Completion:
            started = time.perf_counter()
            resp = await self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            elapsed = int((time.perf_counter() - started) * 1000)
            usage = resp.usage
            return Completion(
                text=(resp.choices[0].message.content or "").strip(),
                model=model,
                latency_ms=elapsed,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )

        async with self._sem:
            out = await _call()
        self.usage.add(role, out)
        return out

    async def complete_many(
        self, prompts: list[str], **kwargs
    ) -> list[Completion | None]:
        """Un echec isole renvoie None plutot que de faire tomber la campagne.

        Une evaluation de 2 500 questions ne doit pas etre perdue parce que
        le modele a produit une sortie non parsable sur la question 1 800 ;
        les None sont comptes et reportes dans le run.
        """

        async def _one(p: str) -> Completion | None:
            try:
                return await self.complete(p, **kwargs)
            except Exception:
                return None

        return await asyncio.gather(*(_one(p) for p in prompts))

    # -- embeddings --------------------------------------------------------

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        @retry(
            stop=stop_after_attempt(self.cfg.llm_max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )
        async def _call(batch: list[str]) -> list[list[float]]:
            resp = await self._client.embeddings.create(model=model, input=batch)
            # L'ordre de sortie n'est pas garanti par la spec : on retrie sur
            # l'index plutot que de faire confiance a la position.
            ordered = sorted(resp.data, key=lambda d: d.index)
            return [d.embedding for d in ordered]

        # Ollama tolere mal les gros lots d'embeddings (charge memoire cote
        # serveur) : on decoupe, et on parallelise les lots plutot que de
        # grossir le lot.
        batch_size = 32
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        async def _guarded(batch: list[str]) -> list[list[float]]:
            async with self._sem:
                return await _call(batch)

        results = await asyncio.gather(*(_guarded(b) for b in batches))
        out: list[list[float]] = []
        for r in results:
            out.extend(r)
        return out

    async def embed_one(self, text: str, *, model: str) -> list[float]:
        return (await self.embed([text], model=model))[0]

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()


async def probe(cfg: Settings | None = None) -> dict[str, object]:
    """Verifie que le serveur d'inference repond et liste ses modeles.

    Appele avant chaque campagne : decouvrir a la question 400 qu'un modele
    n'est pas provisionne coute une heure de GPU.
    """
    cfg = cfg or default_settings
    client = AsyncOpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key, timeout=10)
    try:
        models = await client.models.list()
        return {"ok": True, "base_url": cfg.llm_base_url, "models": [m.id for m in models.data]}
    except Exception as exc:  # noqa: BLE001 - diagnostic, on veut le message brut
        return {"ok": False, "base_url": cfg.llm_base_url, "error": str(exc)}
    finally:
        await client.close()
