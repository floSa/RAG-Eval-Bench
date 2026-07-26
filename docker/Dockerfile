# Image unique pour l'UI et le runner : meme code, deux points d'entree.
# Avoir deux images divergerait tot ou tard et rendrait les resultats du
# tableau de bord non reproductibles par le runner.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Couche de dependances separee du code : un changement de source ne
# reinstalle pas l'environnement.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra ui

COPY src/ ./src/
COPY configs/ ./configs/
RUN uv sync --frozen --extra ui

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["ragbench", "doctor"]
