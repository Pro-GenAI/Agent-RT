FROM node:22-slim AS build
WORKDIR /build
COPY package.json package-lock.json tsconfig.json ./
RUN npm ci --ignore-scripts
COPY src ./src
RUN npm run build && npm pack --silent && mv agent-rt-*.tgz agent-rt.tgz

FROM node:22-slim
ARG AGENT_RT_VERSION=0.1.3
LABEL org.opencontainers.image.version="${AGENT_RT_VERSION}"
WORKDIR /app
COPY --from=build /build/agent-rt.tgz /build/package.json /tmp/
COPY scripts/prune-docker-deps.mjs /tmp/prune-docker-deps.mjs
# Only the commonly needed peers (OpenAI and Anthropic SDKs), at their declared ranges.
# Heavy optional peers (sandboxes, tracing) are left out.
# --ignore-scripts skips onnxruntime-node's postinstall, which downloads CUDA packages from NuGet and fails.
RUN npm init -y >/dev/null \
 && node -e 'const {execFileSync:x}=require("child_process");const p=require("/tmp/package.json").peerDependencies;x("npm",["install","--omit=dev","--omit=optional","--ignore-scripts","--no-audit","--no-fund","/tmp/agent-rt.tgz",...["openai","@anthropic-ai/sdk"].map(k=>k+"@"+p[k])],{stdio:"inherit"})' \
 && node /tmp/prune-docker-deps.mjs /app/node_modules \
 && rm -rf /tmp/* ~/.npm
USER node
CMD ["node"]
