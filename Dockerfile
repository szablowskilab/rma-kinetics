FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

COPY src/ src/

RUN uv venv && uv pip install -e .

COPY app/main.py app/main.py

EXPOSE 8080

CMD ["uv", "run", "marimo", "run", "app/main.py", "--host", "0.0.0.0", "-p", "8080", "--headless"]
