"""Approve or deny agent actions from a chat message or phone notification."""

from __future__ import annotations

import logging
import re
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable
from itertools import count
from typing import Any

from ext.runtime.authorization.approvals import ApprovalManager, ApprovalRequest

_logger = logging.getLogger(__name__)

# Chat transport: (notification text, short reply code) -> None.
ChatSender = Callable[[str, str], None]

_VERBS = {
    "write": "edit",
    "edit": "edit",
    "update": "edit",
    "patch": "edit",
    "modify": "edit",
    "delete": "delete",
    "remove": "delete",
    "rm": "delete",
    "run": "run",
    "exec": "run",
    "execute": "run",
    "send": "send",
    "create": "create",
}
_FILE_KEYS = ("path", "file", "filename", "file_path")
_TARGET_KEYS = (*_FILE_KEYS, "url", "command", "name", "to")
_YES = {"approve", "approved", "allow", "yes", "y", "ok"}
_NO = {"deny", "denied", "reject", "no", "n"}


def describe_request(request: ApprovalRequest) -> str:
    """One-line phone-friendly prompt, e.g. ``Agent wants to edit README, approve?``."""
    words = [w for w in re.split(r"[_\-.\s]+", request.call.name.lower()) if w]
    verb = next((_VERBS[w] for w in words if w in _VERBS), " ".join(words))
    key = next((k for k in _TARGET_KEYS if request.call.arguments.get(k)), None)
    target = str(request.call.arguments[key]) if key else ""
    if key in _FILE_KEYS:
        name = re.split(r"[\\/]", target.rstrip("\\/"))[-1]
        target = name.rsplit(".", 1)[0] or name
    # Arguments come from the model: one line only, so they cannot fake extra text.
    target = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", target).split())
    if len(target) > 60:
        target = target[:57] + "..."
    return f"Agent wants to {' '.join(filter(None, (verb, target)))}, approve?"


class ChatApprovals:
    """Pushes approval requests to a chat channel and resolves them from replies.

    Attaches itself to ``manager`` so every request that needs a human is sent
    once with a short reply code; ``handle_reply`` turns "approve"/"deny"
    (plus the code when several are waiting) into a ``once`` grant bound to the
    exact call arguments. Whoever can post replies can approve, so the chat
    transport must only accept messages from the approving human.
    """

    def __init__(self, manager: ApprovalManager, send: ChatSender) -> None:
        self.manager = manager
        self.send = send
        self.pending: dict[str, ApprovalRequest] = {}
        self._codes = count(1)
        manager.notifier = self.notify

    def notify(self, request: ApprovalRequest) -> None:
        # Resumed runs re-check the same call; send it once.
        if any(waiting.id == request.id for waiting in self.pending.values()):
            return
        code = str(next(self._codes))
        self.pending[code] = request
        try:
            self.send(describe_request(request), code)
        except Exception:  # a dead channel must not turn a pause into a crash
            _logger.warning("approval notification failed", exc_info=True)

    def handle_reply(self, text: str, *, actor_id: str = "chat") -> str:
        """Resolve a reply; returns the confirmation to send back to the chat."""
        parts = text.strip().lower().split()
        word = parts[0] if parts else ""
        if word not in _YES | _NO:
            return "Reply 'approve' or 'deny'."
        if len(parts) > 1:
            code = parts[1].lstrip("#")
        elif len(self.pending) == 1:
            code = next(iter(self.pending))
        else:
            return "Several approvals are waiting: reply 'approve <code>' or 'deny <code>'."
        request = self.pending.pop(code, None)
        if request is None:
            return "No such pending approval."
        decision = "allow" if word in _YES else "deny"
        self.manager.resolve(request, decision, actor_id=actor_id)
        summary = describe_request(request).removesuffix(", approve?")
        return ("Approved: " if decision == "allow" else "Denied: ") + summary


def ntfy_sender(
    topic: str, *, server: str = "https://ntfy.sh", token: str | None = None
) -> ChatSender:
    """Phone push through ntfy (https://ntfy.sh); the topic name acts as a secret.

    Posts from a background thread so a slow server never stalls the agent loop.
    """
    url = f"{server.rstrip('/')}/{urllib.parse.quote(topic, safe='')}"
    headers: dict[str, Any] = {"Title": "Agent approval", "Priority": "high"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def post(body: bytes) -> None:
        try:
            request = urllib.request.Request(url, data=body, headers=headers)
            urllib.request.urlopen(request, timeout=10).close()
        except Exception:
            _logger.warning("ntfy approval notification failed", exc_info=True)

    def send(text: str, code: str) -> None:
        body = f"{text}\nReply: approve {code} / deny {code}".encode()
        threading.Thread(target=post, args=(body,), daemon=True).start()

    return send
