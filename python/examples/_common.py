import os

from agent_rt import AgentConfig, ContentPart, ModelMessage, ModelSettings


def configured_model() -> str:
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("ANTHROPIC_MODEL")
    if not model:
        raise RuntimeError("Set OPENAI_MODEL or ANTHROPIC_MODEL before running this example.")
    return model


def make_agent(name: str, instructions: str) -> AgentConfig:
    return AgentConfig(
        name=name,
        instructions=instructions,
        model=ModelSettings(model=configured_model()),
    )


def user_message(text: str) -> ModelMessage:
    return ModelMessage(
        role="user",
        content=(ContentPart(type="text", text=text),),
    )


def response_text(response) -> str:
    if response is None:
        return ""
    return "".join(
        part.text or ""
        for part in response.message.content
        if part.type == "text"
    )
