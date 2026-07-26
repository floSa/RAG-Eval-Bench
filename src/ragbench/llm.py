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

import httpx
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
    # Raisonnement du modele, quand il est expose. Compte dans
    # completion_tokens : un banc qui ignore ces tokens sous-estime son cout
    # d'un ordre de grandeur sur les modeles a raisonnement.
    thinking: str = ""

    @property
    def is_empty(self) -> bool:
        """Reponse vide alors qu'un appel a bien eu lieu.

        Symptome typique d'un budget max_tokens epuise par le raisonnement.
        A compter separement des erreurs : l'appel a reussi, c'est le
        resultat qui est inexploitable.
        """
        return not self.text.strip()


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
        # Echappatoire assumee vers l'API native d'Ollama, utilisee UNIQUEMENT
        # quand une config demande explicitement d'activer ou de desactiver le
        # raisonnement. L'endpoint OpenAI-compatible d'Ollama ignore le champ
        # `think` sans le signaler, tout en facturant les tokens de reflexion :
        # mesure faite sur gemma4:e4b, 499 tokens generes avec raisonnement
        # contre 7 sans, pour la meme reponse.
        # Tout le reste du banc continue de passer par /v1, donc vLLM et les
        # frameworks d'evaluation fonctionnent sans changement — ils n'ont
        # simplement pas acces a ce reglage.
        base = self.cfg.llm_base_url.rstrip("/")
        self._ollama_native = base[: -len("/v1")] if base.endswith("/v1") else None
        self._http = httpx.AsyncClient(timeout=self.cfg.llm_timeout_s)

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
        thinking: bool | None = None,
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if thinking is not None and self._ollama_native:
            out = await self._complete_ollama_native(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                thinking=thinking,
            )
            self.usage.add(role, out)
            return out

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

    async def _complete_ollama_native(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        thinking: bool,
    ) -> Completion:
        """Appel via /api/chat, seul endpoint qui honore `think`."""

        @retry(
            stop=stop_after_attempt(self.cfg.llm_max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )
        async def _call() -> Completion:
            started = time.perf_counter()
            resp = await self._http.post(
                f"{self._ollama_native}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "think": thinking,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        **({"seed": seed} if seed is not None else {}),
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message", {})
            return Completion(
                text=(message.get("content") or "").strip(),
                model=model,
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt_tokens=data.get("prompt_eval_count", 0) or 0,
                completion_tokens=data.get("eval_count", 0) or 0,
                thinking=(message.get("thinking") or "").strip(),
            )

        async with self._sem:
            return await _call()

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
        await self._http.aclose()

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
