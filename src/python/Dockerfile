FROM python:3.12-slim
ARG AGENT_RT_VERSION=0.1.2
LABEL org.opencontainers.image.version="${AGENT_RT_VERSION}"
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-compile ".[providers,api,web]" && rm -rf /build
RUN useradd -m agent
USER agent
WORKDIR /app
ENTRYPOINT ["agent-rt"]
