from __future__ import annotations

from agent_rt.agents import Agent as OpenAIAgent
from agent_rt.agents import Runner, function_tool
from agent_rt.autogen_agentchat.agents import AssistantAgent
from agent_rt.autogen_agentchat.conditions import TextMentionTermination
from agent_rt.autogen_agentchat.teams import RoundRobinGroupChat
from agent_rt.autogen_ext.models.openai import OpenAIChatCompletionClient
from agent_rt.crewai import Agent as CrewAgent
from agent_rt.crewai import Crew, Process, Task

from agent_rt import ContentPart, ModelMessage, ModelResponse, ToolCall


class QueueProvider:
    name = "queue"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected model request")
        return self.responses.pop(0)


def response(text="", *, tool_calls=(), finish_reason="stop"):
    return ModelResponse(
        message=ModelMessage(
            role="assistant",
            content=(ContentPart(type="text", text=text),) if text else (),
            tool_calls=tuple(tool_calls),
        ),
        model="test-model",
        finish_reason=finish_reason,
    )


async def test_openai_agents_compat_runs_tools_through_agent_rt():
    provider = QueueProvider(
        [
            response(
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="lookup",
                        arguments={"topic": "runtime"},
                    ),
                ),
                finish_reason="tool_calls",
            ),
            response("Agent RT result"),
        ]
    )
    calls = []

    @function_tool
    def lookup(topic: str) -> str:
        """Look up a topic."""
        calls.append(topic)
        return f"found {topic}"

    agent = OpenAIAgent(
        name="Researcher",
        instructions="Use tools when useful.",
        model="test-model",
        provider=provider,
        tools=[lookup],
    )
    result = await Runner.run(agent, "Research runtime")

    assert result.final_output == "Agent RT result"
    assert result.last_agent is agent
    assert calls == ["runtime"]
    assert provider.requests[0].tools[0].name == "lookup"


async def test_autogen_agentchat_compat_single_agent_and_round_robin():
    provider = QueueProvider([response("TERMINATE"), response("TERMINATE")])
    client = OpenAIChatCompletionClient(model="test-model", provider=provider)
    assistant = AssistantAgent(
        "assistant",
        model_client=client,
        system_message="Answer briefly.",
    )

    single = await assistant.run(task="Say done")
    assert single.messages[-1].content == "TERMINATE"

    team = RoundRobinGroupChat(
        [assistant],
        termination_condition=TextMentionTermination("TERMINATE"),
        max_turns=2,
    )
    grouped = await team.run(task="Finish the task")
    assert grouped.messages[-1].source == "assistant"
    assert "TERMINATE" in grouped.stop_reason


async def test_crewai_compat_sequential_tasks_route_through_agent_rt():
    provider = QueueProvider([response("research"), response("final")])
    researcher = CrewAgent(
        role="Researcher",
        goal="Find facts",
        llm="test-model",
        provider=provider,
    )
    writer = CrewAgent(
        role="Writer",
        goal="Write the answer",
        llm="test-model",
        provider=provider,
    )
    research = Task(
        description="Research {topic}",
        expected_output="Useful notes",
        agent=researcher,
    )
    writing = Task(
        description="Write about {topic}",
        expected_output="Final answer",
        agent=writer,
        context=[research],
    )
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research, writing],
        process=Process.sequential,
    )

    result = await crew.kickoff_async(inputs={"topic": "Agent RT"})

    assert result.raw == "final"
    assert [item.raw for item in result.tasks_output] == ["research", "final"]
    assert any(
        "Context from prior tasks" in (part.text or "")
        for message in provider.requests[1].messages
        for part in message.content
    )


def test_framework_compat_exact_prefix_modules_import():
    from agent_rt.agents import Agent
    from agent_rt.autogen_agentchat.agents import AssistantAgent as ImportedAssistant
    from agent_rt.autogen_ext.models.openai import (
        OpenAIChatCompletionClient as ImportedClient,
    )
    from agent_rt.crewai import Crew as ImportedCrew
    from agent_rt.openai_agents import Agent as OpenAIAliasAgent

    assert Agent is OpenAIAgent
    assert OpenAIAliasAgent is OpenAIAgent
    assert ImportedAssistant is AssistantAgent
    assert ImportedClient is OpenAIChatCompletionClient
    assert ImportedCrew is Crew
