FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install .[dev]

EXPOSE 11434

CMD ["ollama-local", "serve", "--host", "0.0.0.0", "--port", "11434"]
