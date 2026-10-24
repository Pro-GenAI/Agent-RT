FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-compile ".[providers,api,web]" && rm -rf /build
RUN useradd -m agent
USER agent
WORKDIR /app
ENTRYPOINT ["agent-rt"]
