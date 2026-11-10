#!/usr/bin/env node
import http from "node:http";
import { WebSocketServer } from "ws";

export const MODEL_ID = "gpt-4o-mini";
export const RESPONSE_TEXT = "mock response";
export const STREAM_PARTS = ["mock ", "response"];
export const TOOL_NAME = "lookup";
export const TOOL_ARGUMENTS = { value: 1 };
export const STRUCTURED_RESPONSE = { value: 1 };

function usage(messages) {
  const prompt = Math.max(
    1,
    Math.floor(
      messages.reduce((sum, item) => sum + String(item?.content ?? "").length, 0) / 4
    )
  );
  const completion = 3;
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: prompt + completion,
  };
}

function json(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

export async function startMockResponsesWebSocket() {
  let responseCounter = 0;
  const server = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  server.on("connection", (socket) => {
    socket.on("message", (raw) => {
      const payload = JSON.parse(raw.toString());
      if (payload.type !== "response.create") {
        socket.send(JSON.stringify({ type: "error", error: { message: "expected response.create" } }));
        return;
      }
      responseCounter += 1;
      const streamId = payload.stream_id;
      const model = payload.model ?? MODEL_ID;
      const input = Array.isArray(payload.input) ? payload.input : [];
      const structured = Boolean(payload.text);
      const text = structured ? JSON.stringify(STRUCTURED_RESPONSE) : RESPONSE_TEXT;
      const inputTokens = Math.max(1, Math.floor(JSON.stringify(input).length / 4));
      const usage = { input_tokens: inputTokens, output_tokens: 3, total_tokens: inputTokens + 3 };
      let output;
      if (
        Array.isArray(payload.tools) && payload.tools.length > 0 &&
        !input.some((item) => item?.type === "function_call_output")
      ) {
        const argumentsText = JSON.stringify(TOOL_ARGUMENTS);
        socket.send(JSON.stringify({
          type: "response.output_item.added",
          stream_id: streamId,
          output_index: 0,
          item: {
            type: "function_call", id: "fc_mock_lookup", call_id: "call_mock_lookup",
            name: TOOL_NAME, arguments: "",
          },
        }));
        socket.send(JSON.stringify({
          type: "response.function_call_arguments.delta",
          stream_id: streamId,
          output_index: 0,
          delta: argumentsText,
        }));
        output = [{
          type: "function_call", id: "fc_mock_lookup", call_id: "call_mock_lookup",
          name: TOOL_NAME, arguments: argumentsText,
        }];
      } else {
        for (const part of structured ? [text] : STREAM_PARTS) {
          socket.send(JSON.stringify({
            type: "response.output_text.delta", stream_id: streamId, delta: part,
          }));
        }
        output = [{
          type: "message", role: "assistant",
          content: [{ type: "output_text", text }],
        }];
      }
      socket.send(JSON.stringify({
        type: "response.completed",
        stream_id: streamId,
        response: {
          id: `resp_mock_${responseCounter}`, model, output, usage,
        },
      }));
    });
  });
  await new Promise((resolve, reject) => {
    server.once("listening", resolve);
    server.once("error", reject);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("Responses WebSocket mock did not expose a TCP address");
  }
  return {
    server,
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
  };
}

export async function startMockOpenAI() {
  const server = http.createServer((req, res) => {
    if (req.method === "GET" && req.url?.replace(/\/$/, "") === "/v1/models") {
      json(res, 200, {
        object: "list",
        data: [{ id: MODEL_ID, object: "model", owned_by: "mock" }],
      });
      return;
    }

    if (req.method !== "POST" || req.url?.replace(/\/$/, "") !== "/v1/chat/completions") {
      json(res, 404, { error: { message: "not found", type: "mock_error" } });
      return;
    }

    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      const payload = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
      const messages = payload.messages ?? [];
      const model = payload.model ?? MODEL_ID;

      if (payload.stream) {
        const created = Math.floor(Date.now() / 1000);
        const events = STREAM_PARTS.map((text, index) => {
          const part = {
            id: "chatcmpl-mock",
            object: "chat.completion.chunk",
            created,
            model,
            choices: [
              {
                index: 0,
                delta: {
                  ...(index === 0 ? { role: "assistant" } : {}),
                  content: text,
                },
                finish_reason: null,
              },
            ],
          };
          return `data: ${JSON.stringify(part)}\n\n`;
        });
        events.push(
          `data: ${JSON.stringify({
            id: "chatcmpl-mock",
            object: "chat.completion.chunk",
            created,
            model,
            choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
            usage: usage(messages),
          })}\n\n`,
          "data: [DONE]\n\n",
        );
        const body = events.join("");
        res.writeHead(200, {
          "content-type": "text/event-stream",
          "cache-control": "no-cache",
          "content-length": Buffer.byteLength(body),
        });
        res.end(body);
        return;
      }

      if (payload.response_format) {
        const latestUser = [...messages].reverse().find((item) => item?.role === "user");
        const content =
          String(latestUser?.content ?? "").includes("[structured-invalid]") &&
          !String(latestUser?.content ?? "").includes("[structured-repair]")
            ? '{"value":"invalid"}'
            : JSON.stringify(STRUCTURED_RESPONSE);
        json(res, 200, {
          id: "chatcmpl-mock",
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [{
            index: 0,
            message: { role: "assistant", content },
            finish_reason: "stop",
          }],
          usage: usage(messages),
        });
        return;
      }

      if (
        Array.isArray(payload.tools) &&
        payload.tools.length > 0 &&
        !messages.some((item) => item?.role === "tool")
      ) {
        json(res, 200, {
          id: "chatcmpl-mock",
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [{
            index: 0,
            message: {
              role: "assistant",
              content: null,
              tool_calls: [{
                id: "call_mock_lookup",
                type: "function",
                function: {
                  name: TOOL_NAME,
                  arguments: JSON.stringify(TOOL_ARGUMENTS),
                },
              }],
            },
            finish_reason: "tool_calls",
          }],
          usage: usage(messages),
        });
        return;
      }

      json(res, 200, {
        id: "chatcmpl-mock",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: RESPONSE_TEXT },
            finish_reason: "stop",
          },
        ],
        usage: usage(messages),
      });
    });
  });

  server.on("connection", (socket) => socket.setNoDelay(true));

  await new Promise((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolveListen);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("mock server did not expose a TCP address");
  }
  return {
    server,
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { server, baseUrl } = await startMockOpenAI();
  console.log(baseUrl);
  const close = () => server.close(() => process.exit(0));
  process.on("SIGINT", close);
  process.on("SIGTERM", close);
}
