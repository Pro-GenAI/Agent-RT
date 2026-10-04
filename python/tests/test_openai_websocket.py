import asyncio
import json

from websockets.asyncio.server import serve

from agent_rt import (
    ContentPart,
    ModelMessage,
    ModelRequest,
    OpenAIModelProvider,
    OpenAIProviderSettings,
    ReasoningConfig,
    ToolDefinition,
)


class _ResponsesWebSocketMock:
    def __init__(self):
        self.connection_count = 0
        self.requests = []

    async def handler(self, websocket):
        self.connection_count += 1
        self.path = websocket.request.path
        self.authorization = websocket.request.headers.get("authorization")
        async for raw in websocket:
            request = json.loads(raw)
            self.requests.append(request)
            stream_id = request["stream_id"]
            response_index = len(self.requests)

            if response_index == 1:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "response.output_item.added",
                            "stream_id": stream_id,
                            "output_index": 0,
                            "item": {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "call_1",
                                "name": "lookup",
                                "arguments": "",
                            },
                        }
                    )
                )
                await websocket.send(
                    json.dumps(
                        {
                            "type": "response.function_call_arguments.delta",
                            "stream_id": stream_id,
                            "output_index": 0,
                            "delta": '{"q":"x"}',
                        }
                    )
                )
                await websocket.send(
                    json.dumps(
                        {
                            "type": "response.completed",
                            "stream_id": stream_id,
                            "response": {
                                "id": "resp_1",
                                "model": "gpt-test",
                                "output": [
                                    {
                                        "type": "function_call",
                                        "id": "fc_1",
                                        "call_id": "call_1",
                                        "name": "lookup",
                                        "arguments": '{"q":"x"}',
                                    }
                                ],
                                "usage": {
                                    "input_tokens": 3,
                                    "output_tokens": 2,
                                    "total_tokens": 5,
                                },
                            },
                        }
                    )
                )
                continue

            await websocket.send(
                json.dumps(
                    {
                        "type": "response.output_text.delta",
                        "stream_id": stream_id,
                        "delta": "done",
                    }
                )
            )
            await websocket.send(
                json.dumps(
                    {
                        "type": "response.completed",
                        "stream_id": stream_id,
                        "response": {
                            "id": "resp_2",
                            "model": "gpt-test",
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [
                                        {"type": "output_text", "text": "done"}
                                    ],
                                }
                            ],
                            "usage": {
                                "input_tokens": 2,
                                "output_tokens": 1,
                                "total_tokens": 3,
                            },
                        },
                    }
                )
            )


class TestOpenAIResponsesWebSocket:
    async def test_provider_defaults_to_persistent_responses_websocket(self):
        mock = _ResponsesWebSocketMock()
        async with serve(mock.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            provider = OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=f"http://127.0.0.1:{port}/v1",
                    api_key="test-key",
                    default_model="gpt-test",
                )
            )
            try:
                user = ModelMessage(
                    role="user",
                    content=(ContentPart(type="text", text="hello"),),
                )
                tool = ToolDefinition(
                    name="lookup",
                    description="Lookup a value",
                    input_schema={
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                )
                first = await provider.complete(
                    ModelRequest(
                        messages=(user,),
                        tools=(tool,),
                        reasoning=ReasoningConfig(effort="high", summary="auto"),
                    )
                )
                assert first.finish_reason == "tool_calls"
                assert first.message.tool_calls[0].id == "call_1"
                assert first.message.tool_calls[0].arguments == {"q": "x"}

                tool_result = ModelMessage(
                    role="tool",
                    content=(ContentPart(type="text", text="result"),),
                    tool_call_id="call_1",
                )
                events = [
                    event
                    async for event in provider.stream(
                        ModelRequest(
                            messages=(user, first.message, tool_result),
                            tools=(tool,),
                        )
                    )
                ]
                assert (
                    "".join(
                        event.text or ""
                        for event in events
                        if event.type == "text_delta"
                    )
                    == "done"
                )
                assert events[-1].response.message.content[0].text == "done"
            finally:
                await provider.close()

        assert mock.connection_count == 1
        assert mock.path == "/v1/responses"
        assert mock.authorization == "Bearer test-key"
        assert len(mock.requests) == 2

        first_request, second_request = mock.requests
        assert first_request["type"] == "response.create"
        assert first_request["model"] == "gpt-test"
        assert not first_request["store"]
        assert "previous_response_id" not in first_request
        assert first_request["input"][0]["role"] == "user"
        assert first_request["tools"][0]["name"] == "lookup"
        assert first_request["reasoning"] == {"effort": "high", "summary": "auto"}

        assert second_request["previous_response_id"] == "resp_1"
        assert second_request["input"] == [
            {"type": "function_call_output", "call_id": "call_1", "output": "result"}
        ]

    async def test_provider_retries_without_stream_id_when_endpoint_rejects_it(self):
        requests = []

        async def handler(websocket):
            request = json.loads(await websocket.recv())
            requests.append(request)
            if len(requests) == 1:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "status": 400,
                            "error": {"detail": "Unsupported parameter: stream_id"},
                        }
                    )
                )
                return

            await websocket.send(
                json.dumps(
                    {
                        "type": "response.output_text.delta",
                        "delta": "done",
                    }
                )
            )
            await websocket.send(
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_legacy",
                            "model": "gpt-test",
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": "done"}],
                                }
                            ],
                        },
                    }
                )
            )

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            provider = OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=f"http://127.0.0.1:{port}/v1",
                    default_model="gpt-test",
                )
            )
            try:
                response = await asyncio.wait_for(
                    provider.complete(
                        ModelRequest(
                            messages=(
                                ModelMessage(
                                    role="user",
                                    content=(ContentPart(type="text", text="hello"),),
                                ),
                            )
                        )
                    ),
                    timeout=2,
                )
            finally:
                await provider.close()

        assert response.message.content[0].text == "done"
        assert "stream_id" in requests[0]
        assert "stream_id" not in requests[1]

    async def test_provider_multiplexes_concurrent_responses_by_stream_id(self):
        requests = []

        async def handler(websocket):
            while len(requests) < 2:
                requests.append(json.loads(await websocket.recv()))
            for index, request in enumerate(reversed(requests), start=1):
                stream_id = request["stream_id"]
                text = request["input"][0]["content"]
                await websocket.send(
                    json.dumps(
                        {
                            "type": "response.output_text.delta",
                            "stream_id": stream_id,
                            "delta": text,
                        }
                    )
                )
                await websocket.send(
                    json.dumps(
                        {
                            "type": "response.completed",
                            "stream_id": stream_id,
                            "response": {
                                "id": f"resp_concurrent_{index}",
                                "model": "gpt-test",
                                "output": [
                                    {
                                        "type": "message",
                                        "role": "assistant",
                                        "content": [
                                            {"type": "output_text", "text": text}
                                        ],
                                    }
                                ],
                            },
                        }
                    )
                )

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            provider = OpenAIModelProvider(
                OpenAIProviderSettings(
                    base_url=f"http://127.0.0.1:{port}/v1",
                    default_model="gpt-test",
                )
            )
            try:
                make_request = lambda text: ModelRequest(
                    messages=(
                        ModelMessage(
                            role="user",
                            content=(ContentPart(type="text", text=text),),
                        ),
                    )
                )
                first, second = await asyncio.wait_for(
                    asyncio.gather(
                        provider.complete(make_request("first")),
                        provider.complete(make_request("second")),
                    ),
                    timeout=2,
                )
            finally:
                await provider.close()

        assert first.message.content[0].text == "first"
        assert second.message.content[0].text == "second"
        assert len({request["stream_id"] for request in requests}) == 2
