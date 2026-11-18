> **Canonical HTML page:** [Install Agent RT for Python & TypeScript — Agent RT Docs](https://agent-rt.pro-genai.com/docs/getting-started/installation)

> Install Agent RT for Python 3.10–3.14 or Node.js 20, 22, 24, and 26, then configure OpenAI, Anthropic, or compatible model providers.

[Docs](https://agent-rt.pro-genai.com/docs/)

/Getting started/Installation
 Getting started

# Installation

Agent RT ships as a Python package and a TypeScript/Node package. Choose the runtime your application already uses; the core contracts intentionally follow the same mental model in both implementations.

## Requirements

| Runtime | Supported versions | Install |
| --- | --- | --- |
| Python | 3.10–3.14 | `pip install agent-rt` |
| Node.js | 20, 22, 24, or 26 | `npm install agent-rt` |

## Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install agent-rt
```

## TypeScript / Node.js

```bash
npm install agent-rt
```

## Docker Hub images

Prebuilt images are also available on Docker Hub:

[Agent RT Python](https://hub.docker.com/r/praneeth1v/agent-rt-python)

and

[Agent RT TypeScript](https://hub.docker.com/r/praneeth1v/agent-rt-typescript)

. Pull the latest tag, or use a published version tag such as `0.1.2`:

```bash
docker pull praneeth1v/agent-rt-python:latest
docker pull praneeth1v/agent-rt-typescript:latest
```

The Python image starts the `agent-rt` CLI; the TypeScript image starts Node.js and has the `agent-rt` npm package preinstalled:

```bash
docker run --rm praneeth1v/agent-rt-python:latest --help
docker run --rm praneeth1v/agent-rt-typescript:latest node -e "console.log(typeof require('agent-rt').AgentLoop)"
```

The Python image includes the API and web extras; the TypeScript image includes the OpenAI and Anthropic SDK peers. Docker Hub displays each image's pull count; the

[GitHub README](https://github.com/Pro-GenAI/Agent-RT/blob/main/README.md)

also includes live pull-count badges. Provide provider credentials/configuration at runtime, not in your image.

## Configure models

Agent RT is configured with one YAML file: `~/.agent-rt/config.yaml` in your home folder, or a project file explicitly passed with `--config` / `load_config(path)`. Declare as many models as you need; secrets are referenced as `os.environ/NAME` and are the only thing read from the environment.

```yaml
model_list:
  - model_name: my-model
    agent_rt_params:
      model: openai/your-model          # or anthropic/...; add api_base for compatible endpoints
      api_key: os.environ/OPENAI_API_KEY
```

Repeating a `model_name` adds an ordered fallback. See the

[configuration reference](https://agent-rt.pro-genai.com/docs/reference/configuration.html)

and the

[example file](https://github.com/Pro-GenAI/Agent-RT/blob/main/agent-rt.example.yaml)

.

 **Provider-neutral by design**Agent code is built against Agent RT contracts. OpenAI, Anthropic, compatible endpoints, Decision models, and custom providers can sit behind that boundary.

## Vector databases

No Chroma, Pinecone, Qdrant, Milvus, Weaviate, LangChain, or LlamaIndex package is required for Agent RT's built-in vector retrieval and compatibility adapters. Declare services in `vector_store_registry` (several are allowed), then see the

[vector database guide](https://agent-rt.pro-genai.com/docs/guides/vector-databases.html)

for backend credentials and framework migration imports.

[PreviousDocumentation overview](https://agent-rt.pro-genai.com/docs/)



[NextQuickstart →](https://agent-rt.pro-genai.com/docs/getting-started/quickstart.html)
