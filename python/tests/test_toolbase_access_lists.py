import pytest

from agent_rt import TOOL_SEARCH_NAME, ToolCall, ToolDefinition, ToolRegistry, Toolbase


@pytest.mark.asyncio
async def test_toolbase_access_lists_hide_search_and_reject_calls():
    registry = ToolRegistry()
    for name, effect in (("read_ok", "none"), ("read_hidden", "none"),
                         ("write_ok", "consequential"), ("write_hidden", "consequential")):
        registry.register(ToolDefinition(
            name=name, description=f"{name} searchable catalog entry",
            input_schema={"type": "object"}, side_effect=effect,
        ), handler=lambda args, token: "executed")

    base = await Toolbase.initialize(
        tool_registry=registry,
        allowlist=("read_ok", "read_hidden", "write_ok", "write_hidden"),
        blocklist=("read_hidden",),
        read_allowlist=("read_ok", "read_hidden"),
        write_allowlist=("write_ok", "write_hidden"),
        write_blocklist=("write_hidden",),
    )
    assert {d.name for d in base.definitions()} == {TOOL_SEARCH_NAME, "read_ok", "write_ok"}
    search = await base.execute(ToolCall(id="search", name=TOOL_SEARCH_NAME, arguments={"query": "searchable catalog entry"}))
    assert {tool["name"] for tool in search["tools"]} <= {"read_ok", "write_ok"}
    for name in ("read_hidden", "write_hidden", "invented"):
        with pytest.raises(KeyError):
            await base.execute(ToolCall(id=name, name=name, arguments={}))


@pytest.mark.asyncio
async def test_toolbase_blocks_mcp_tools_before_registration():
    class Remote:
        async def list_tools(self):
            return [{"name": "blocked", "input_schema": {"type": "object"}}]
        async def call_tool(self, name, arguments):
            raise AssertionError("blocked remote handler invoked")

    registry = ToolRegistry()
    base = await Toolbase.initialize(tool_registry=registry, mcp_clients={"remote": Remote()},
                                     blocklist=("mcp.remote.blocked",))
    assert {d.name for d in base.definitions()} == {TOOL_SEARCH_NAME}
    with pytest.raises(KeyError):
        registry.get("mcp.remote.blocked")

