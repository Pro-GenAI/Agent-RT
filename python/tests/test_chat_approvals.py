from __future__ import annotations

import pytest

from agent_rt import (
    ApprovalDeniedError,
    ApprovalManager,
    ApprovalRequest,
    ApprovalRequiredError,
    ChatApprovals,
    ToolCall,
    ToolDefinition,
)
from ext.runtime.authorization.chat_approvals import describe_request, ntfy_sender


def _definition(name: str = "write_file") -> ToolDefinition:
    return ToolDefinition(
        name=name, description="d", input_schema={}, side_effect="consequential"
    )


def _setup():
    sent: list[tuple[str, str]] = []
    manager = ApprovalManager()
    chat = ChatApprovals(manager, lambda text, code: sent.append((text, code)))
    return sent, manager, chat


def _describe(name: str, **arguments: object) -> str:
    call = ToolCall(id="c", name=name, arguments=arguments)
    return describe_request(ApprovalRequest("a", call, "consequential", "r"))


class TestChatApprovals:
    def test_notifies_once_and_approves_by_reply(self) -> None:
        sent, manager, chat = _setup()
        call = ToolCall(
            id="c1", name="write_file", arguments={"path": "docs/README.md"}
        )
        for _ in range(2):  # a resumed run re-checks the same call
            with pytest.raises(ApprovalRequiredError):
                manager.check(call, _definition())
        assert sent == [(_describe("write_file", path="docs/README.md"), "1")]
        assert (
            chat.handle_reply("Approve")
            == 'Approved: Agent requests {"arguments":{"path":"docs/README.md"},"tool":"write_file"}'
        )
        manager.check(call, _definition())

    def test_codes_pick_among_several_and_deny(self) -> None:
        sent, manager, chat = _setup()
        calls = [
            ToolCall(id=f"Call_{n}", name="write_file", arguments={"path": f"{n}.md"})
            for n in "AB"
        ]
        for call in calls:
            with pytest.raises(ApprovalRequiredError):
                manager.check(call, _definition())
        assert [code for _, code in sent] == ["1", "2"]
        assert chat.handle_reply("maybe").startswith("Reply")
        assert chat.handle_reply("yes").startswith("Several")
        assert (
            chat.handle_reply("deny #2")
            == 'Denied: Agent requests {"arguments":{"path":"B.md"},"tool":"write_file"}'
        )
        assert chat.handle_reply("deny 2") == "No such pending approval."
        with pytest.raises(ApprovalDeniedError):
            manager.check(calls[1], _definition())
        assert (
            chat.handle_reply("y")
            == 'Approved: Agent requests {"arguments":{"path":"A.md"},"tool":"write_file"}'
        )

    def test_failing_channel_still_pauses(self) -> None:
        manager = ApprovalManager()

        def boom(text: str, code: str) -> None:
            raise OSError("offline")

        ChatApprovals(manager, boom)
        with pytest.raises(ApprovalRequiredError):
            manager.check(ToolCall(id="1", name="rm", arguments={}), _definition("rm"))

    def test_description_is_one_safe_line(self) -> None:
        import json

        arguments = {
            "path": "docs/README.md",
            "command": "echo safe\nrm -rf /srv/data\x1b[2K\u202e",
        }
        prompt = _describe("run_shell", **arguments)
        assert "\n" not in prompt and "\x1b" not in prompt and "\u202e" not in prompt
        assert json.loads(
            prompt.removeprefix("Agent requests ").removesuffix(", approve?")
        ) == {"tool": "run_shell", "arguments": arguments}
        assert _describe("edit", path="a/README.md") != _describe(
            "edit", path="b/README.py"
        )
        assert "hidden-tail" in _describe(
            "run_shell", command="x" * 5000 + "hidden-tail"
        )


def test_ntfy_sender_rejects_non_http_server() -> None:
    with pytest.raises(ValueError):
        ntfy_sender("topic", server="file:///etc")
