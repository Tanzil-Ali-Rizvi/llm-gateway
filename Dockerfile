FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LLMGW_CONFIG=config/gateway.yaml LLMGW_DATABASE=/data/gateway.db
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd -m gateway && mkdir /data && chown gateway /data
COPY config ./config
USER gateway
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["sh", "-c", "uvicorn 'llmgw.app:create_app' --factory --host 0.0.0.0 --port 8080"]
