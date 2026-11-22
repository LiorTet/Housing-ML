FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY src ./src
COPY config ./config
RUN uv sync --frozen --no-dev

ENV ENV=prod PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "housing.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
