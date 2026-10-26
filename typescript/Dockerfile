FROM node:22-slim AS build
WORKDIR /build
COPY package.json package-lock.json tsconfig.json ./
RUN npm ci --ignore-scripts
COPY src ./src
RUN npm run build && npm pack --silent && mv agent-rt-*.tgz agent-rt.tgz

FROM node:22-slim
WORKDIR /app
COPY --from=build /build/agent-rt.tgz /build/package.json /tmp/
# Only the commonly needed peers (OpenAI and Anthropic SDKs), at their declared ranges.
# Heavy optional peers (sandboxes, tracing) are left out.
RUN npm init -y >/dev/null \
 && node -e 'const {execFileSync:x}=require("child_process");const p=require("/tmp/package.json").peerDependencies;x("npm",["install","--no-audit","--no-fund","/tmp/agent-rt.tgz",...["openai","@anthropic-ai/sdk"].map(k=>k+"@"+p[k])],{stdio:"inherit"})' \
 && rm -rf /tmp/* ~/.npm
USER node
CMD ["node"]
