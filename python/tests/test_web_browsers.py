"""Real-browser checks for the web UI. Run through `make test-browsers`, which picks the browser.

Skipped in the normal suite: they need Playwright and AGENT_RT_BROWSER (`chromium`, `firefox`,
or `webkit`). AGENT_RT_CHROMIUM_PATH runs `chromium` against your own executable.
"""

import asyncio
import os
import threading
import time

import pytest

from agent_rt import (
    AgentLoop,
    ContentPart,
    ModelMessage,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ToolCall,
)
from agent_rt_cli import ChatSession, TranscriptStore, WorkspaceCodeTools
from agent_rt_web import WebChat, create_web_app

TARGET = os.environ.get("AGENT_RT_BROWSER", "")
if not TARGET:
    pytest.skip("browser tests run through make test-browsers", allow_module_level=True)
sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect
uvicorn = pytest.importorskip("uvicorn")

REPLY = "Here is a **plan**:\n\n- read `main.py`\n\n```python\nx = 1\n```\n\n<script>alert(1)</script>"


class DemoProvider:
    """Writes a file when asked to "write", otherwise streams a Markdown reply."""

    name = "demo"

    async def complete(self, request):
        last = request.messages[-1]
        prompt = next(m for m in reversed(request.messages) if m.role == "user")
        if "steps" in (prompt.content[0].text or ""):
            # Text, a read-only tool call, then more text: two steps of one run.
            if last.role == "tool":
                message = ModelMessage(
                    role="assistant",
                    content=(ContentPart(type="text", text="Answer."),),
                )
            else:
                call = ToolCall(id="s1", name="list_files", arguments={"path": "."})
                text = (ContentPart(type="text", text="Looking."),)
                message = ModelMessage(
                    role="assistant", content=text, tool_calls=(call,)
                )
        elif last.role == "user" and "write" in (last.content[0].text or ""):
            call = ToolCall(
                id="c1",
                name="write_file",
                arguments={"path": "hello.txt", "content": "hi"},
            )
            message = ModelMessage(role="assistant", content=(), tool_calls=(call,))
        else:
            prompt_text = last.content[0].text or ""
            text = (
                "x" * 50000
                if "unbroken" in prompt_text
                else REPLY * 300 if "long" in prompt_text else REPLY
            )
            message = ModelMessage(
                role="assistant", content=(ContentPart(type="text", text=text),)
            )
        return ModelResponse(
            message=message,
            model=request.model,
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason="stop",
        )

    async def stream(self, request):
        response = await self.complete(request)
        for part in response.message.content:
            for start in range(0, len(part.text), 16):
                await asyncio.sleep(0.01 if len(part.text) < 1000 else 0)
                yield ModelStreamEvent(
                    type="text_delta", text=part.text[start : start + 16]
                )
        for call in response.message.tool_calls:
            for _ in range(3):  # real providers repeat the name on every argument chunk
                yield ModelStreamEvent(
                    type="tool_call_delta", tool_name=call.name, tool_call_id=call.id
                )
        yield ModelStreamEvent(type="completed", response=response)


@pytest.fixture
def server(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def make_session(model, approve):
        tools = WorkspaceCodeTools(workspace)
        tools.approval_callback = approve
        registry = tools.registry(
            tool_input_guardrail_classifier=lambda _r: (None, 0.01)
        )
        return ChatSession(
            AgentLoop(DemoProvider(), tool_executor=registry, tool_registry=registry),
            model=model,
            autosave=False,
            workspace_tools=tools,
        )

    chat = WebChat(
        make_session,
        model="demo-model",
        models=("demo-large",),
        workspace=workspace,
        store=TranscriptStore(tmp_path / "sessions"),
    )
    instance = uvicorn.Server(
        uvicorn.Config(
            create_web_app(chat), host="127.0.0.1", port=0, log_level="warning"
        )
    )
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not instance.started:
        assert time.monotonic() < deadline, "web server did not start"
        time.sleep(0.05)
    port = instance.servers[0].sockets[0].getsockname()[1]
    # The launch link: it signs the browser context in, then redirects to "/" (keeping any #hash).
    yield f"http://127.0.0.1:{port}/?token={chat.token}", workspace
    instance.should_exit = True
    thread.join(10)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        path = os.environ.get("AGENT_RT_CHROMIUM_PATH") if TARGET == "chromium" else None
        launched = getattr(playwright, TARGET).launch(executable_path=path or None)
        yield launched
        launched.close()


def send(page, text):
    page.fill("#input", text)
    page.press("#input", "Enter")


def wait_idle(page, timeout=30_000):
    # While a run is going the send button turns into a stop button.
    page.wait_for_selector("#send:not(.stop)", timeout=timeout)


def test_streams_escapes_approves_and_keeps_session_after_reload(browser, server):
    url, workspace = server
    context = browser.new_context()
    page = context.new_page()
    dialogs = []
    page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
    page.goto(url)
    page.wait_for_selector(".empty")
    # An unsent session has no address and no sidebar entry; the first message names it.
    assert "#" not in page.url
    expect(page.locator("#sessions li")).to_have_count(0)

    send(page, "explain the plan")
    page.wait_for_selector(".turn.assistant pre code")
    expect(page.locator("#sid")).to_have_text("explain the plan")
    assert "#" in page.url
    wait_idle(page)
    body = page.locator(".turn.assistant .body").first
    assert "<script>alert(1)</script>" in body.inner_text()
    assert body.locator("strong").inner_text() == "plan"

    send(page, "write a file")
    page.wait_for_selector(".approval .btn.go")
    page.click(".approval .btn.go")
    wait_idle(page)
    expect(page.locator(".turn")).to_have_count(4)
    assert (workspace / "hello.txt").read_text() == "hi"

    page.reload()
    expect(page.locator(".turn")).to_have_count(4)
    assert page.locator("#sessions .open").first.inner_text() == "explain the plan"
    assert dialogs == []
    context.close()


def test_second_tab_queues_behind_a_running_turn_in_the_same_session(browser, server):
    url, workspace = server
    context = browser.new_context()
    first = context.new_page()
    first.goto(url)
    first.wait_for_selector(".empty")
    send(first, "write a file")
    first.wait_for_selector(".approval .btn.go")

    second = context.new_page()
    second.goto(first.url)
    second.wait_for_selector("#sessions li.on")
    send(second, "explain")
    expect(
        second.get_by_text("waiting for another tab's run in this session")
    ).to_be_visible()

    first.click(".approval .btn.go")
    wait_idle(first)
    wait_idle(second)
    expect(second.locator(".turn")).to_have_count(4)
    users = second.locator(".turn.user .body").all_inner_texts()
    assert users == ["write a file", "explain"]
    assert (workspace / "hello.txt").exists()
    context.close()


def test_phone_width_light_theme_and_sidebar_toggle(browser, server):
    url, _ = server
    context = browser.new_context(
        viewport={"width": 375, "height": 812}, color_scheme="light"
    )
    page = context.new_page()
    page.goto(url)
    page.wait_for_selector(".empty")
    assert page.evaluate("document.documentElement.scrollWidth") <= 375
    background = page.evaluate("getComputedStyle(document.body).backgroundColor")
    assert background == "rgb(243, 241, 234)"
    # Locator waits instead of wait_for_function: the page's CSP rightly blocks eval.
    rail = page.locator("#rail")
    expect(rail).not_to_be_in_viewport()
    page.click("#menu")
    expect(rail).to_be_in_viewport()
    page.keyboard.press("Escape")
    expect(rail).not_to_be_in_viewport()
    # Touch phones have no Escape and the open drawer covers the menu button: a tap beside it closes it.
    page.click("#menu")
    expect(rail).to_be_in_viewport()
    page.mouse.click(340, 400)
    expect(rail).not_to_be_in_viewport()
    context.close()


def test_long_reply_streams_without_freezing_the_page(browser, server):
    url, _ = server
    context = browser.new_context()
    page = context.new_page()
    page.goto(url)
    page.wait_for_selector(".empty")
    started = time.monotonic()
    send(page, "a long answer please")
    # About 30 KB in 16-character chunks; redrawing everything per chunk took minutes.
    wait_idle(page, timeout=20_000)
    assert time.monotonic() - started < 20
    expect(page.locator(".turn.assistant pre")).to_have_count(300)
    context.close()


def test_unbroken_long_block_remains_responsive_and_formats_on_completion(
    browser, server
):
    url, _ = server
    context = browser.new_context()
    page = context.new_page()
    page.goto(url)
    page.wait_for_selector(".empty")
    started = time.monotonic()
    send(page, "unbroken paragraph")
    wait_idle(page, timeout=20_000)
    assert time.monotonic() - started < 20
    expect(page.locator(".turn.assistant .body p")).to_have_count(1)
    assert len(page.locator(".turn.assistant .body").inner_text()) == 50000
    page.reload()
    assert len(page.locator(".turn.assistant .body").inner_text()) == 50000
    context.close()


def test_incremental_markdown_matches_a_full_render(browser, server):
    url, _ = server
    context = browser.new_context()
    page = context.new_page()
    page.goto(url)
    page.wait_for_selector(".empty")
    texts = [
        "Para **b**.\n\n- a\n- b\n\n```py\nx = 1\n\n\ny = 2\n```\n\nEnd.\n\n\n# H\n\nlast",
        "```\nunclosed\n\nfence",
    ]
    mismatches = page.evaluate(
        """(texts) => {
          const bad = [];
          for (const text of texts) for (let step = 1; step <= 7; step++) {
            let html = "", done = 0;
            const scan = newBlockScanner();
            for (let n = step; n < text.length + step; n += step) {
              const raw = text.slice(0, n), cut = scan(raw);
              if (cut > done) { html += md(raw.slice(done, cut)); done = cut; }
            }
            const cut = scan(text, true);
            if (cut > done) { html += md(text.slice(done, cut)); done = cut; }
            if (html + md(text.slice(done)) !== md(text)) bad.push([text, step]);
          }
          return bad;
        }""",
        texts,
    )
    assert page.evaluate("(t) => md(t).includes('<pre>')", texts[0])
    # Items split by blank lines keep their numbers.
    assert (
        page.evaluate("md('1. a\\n\\n2. b')")
        == '<ol start="1"><li>a</li></ol><ol start="2"><li>b</li></ol>'
    )
    assert mismatches == []
    context.close()


def test_terminal_sessions_list_open_and_web_sessions_reach_the_store(browser, server):
    url, workspace = server
    store = TranscriptStore(workspace.parent / "sessions")
    prompt = ModelMessage(
        role="user", content=(ContentPart(type="text", text="asked in the terminal"),)
    )
    store.save(
        "cli1",
        model="demo-model",
        instructions="i",
        messages=(prompt,),
        workspace=str(workspace),
    )
    context = browser.new_context()
    page = context.new_page()
    page.goto(url)
    page.click("#sessions .open:has-text('asked in the terminal')")
    expect(page.locator(".turn.user", has_text="asked in the terminal")).to_be_visible()
    send(page, "explain")
    wait_idle(page)
    assert len(store.load("cli1")["messages"]) > 1
    context.close()


def test_deleting_active_session_waits_for_run_and_does_not_revive_it(browser, server):
    url, _ = server
    context = browser.new_context()
    first = context.new_page()
    first.goto(url)
    first.wait_for_selector(".empty")
    send(first, "write a file")
    first.wait_for_selector(".approval .btn.go")
    second = context.new_page()
    second.goto(first.url)
    second.wait_for_selector("#sessions li.on")
    second.on("dialog", lambda dialog: dialog.accept())
    second.click("#sessions li.on .del")
    first.click(".approval .btn.go")
    wait_idle(first)
    expect(second.locator(".empty")).to_be_visible()
    first.reload()
    expect(first.locator(".empty")).to_be_visible()
    context.close()


def test_steps_of_one_run_stay_separate_and_tools_show_once(browser, server):
    url, _ = server
    context = browser.new_context()
    page = context.new_page()
    page.goto(url)
    page.wait_for_selector(".empty")
    send(page, "two steps please")
    wait_idle(page)
    agent = page.locator(".turn.assistant")
    expect(agent.locator(".tools span")).to_have_count(1)
    live = agent.locator(".body p").all_inner_texts()
    assert live == ["Looking.", "Answer."]
    page.reload()
    expect(agent.locator(".body p")).to_have_count(2)
    assert agent.locator(".body p").all_inner_texts() == live
    context.close()


def test_stop_cancels_the_run_and_its_pending_edit(browser, server):
    url, workspace = server
    context = browser.new_context()
    page = context.new_page()
    page.goto(url)
    page.wait_for_selector(".empty")
    send(page, "write a file")
    page.wait_for_selector(".approval .btn.go")
    expect(page.locator("#send")).to_have_class("stop")
    page.click("#send")
    expect(page.get_by_text("stopped", exact=True)).to_be_visible()
    wait_idle(page)
    expect(page.locator(".approval .btn")).to_have_count(0)
    assert not (workspace / "hello.txt").exists()
    # The page still works after a stop.
    send(page, "explain")
    expect(page.locator(".turn.assistant pre code")).not_to_have_count(0)
    wait_idle(page)
    context.close()


def test_link_to_another_workspaces_session_opens_a_new_one(browser, server):
    url, workspace = server
    secret = ModelMessage(
        role="user", content=(ContentPart(type="text", text="secret plan"),)
    )
    TranscriptStore(workspace.parent / "sessions").save(
        "elsewhere1",
        model="demo-model",
        instructions="i",
        messages=(secret,),
        workspace="/some/other/folder",
    )
    context = browser.new_context()
    page = context.new_page()
    page.goto(url + "#elsewhere1")
    page.wait_for_selector(".empty")
    expect(page.get_by_text("secret plan")).to_have_count(0)
    context.close()


def test_sidebar_resizes_by_drag_and_keys_and_keeps_its_width(browser, server):
    url, _ = server
    context = browser.new_context(viewport={"width": 1200, "height": 800})
    page = context.new_page()
    page.goto(url)
    page.wait_for_selector(".empty")
    handle = page.locator("#resize").bounding_box()
    y = handle["y"] + 100
    page.mouse.move(handle["x"] + handle["width"] / 2, y)
    page.mouse.down()
    page.mouse.move(400, y, steps=5)
    page.mouse.up()
    expect(page.locator("#resize")).to_have_attribute("aria-valuenow", "400")

    page.reload()
    page.wait_for_selector(".empty")
    assert round(page.locator("#rail").bounding_box()["width"]) == 400

    page.focus("#resize")
    page.keyboard.press("ArrowLeft")
    expect(page.locator("#resize")).to_have_attribute("aria-valuenow", "384")
    handle = page.locator("#resize").bounding_box()  # the edge moved
    page.mouse.move(handle["x"] + handle["width"] / 2, y)
    page.mouse.down()
    page.mouse.move(10, y, steps=3)  # clamped to the minimum
    page.mouse.up()
    expect(page.locator("#resize")).to_have_attribute("aria-valuenow", "180")
    context.close()
