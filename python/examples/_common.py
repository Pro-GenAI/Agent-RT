from functools import lru_cache

from agent_rt import (
    AgentConfig,
    ContentPart,
    ModelMessage,
    ModelSettings,
    load_config,
    load_model,
)


@lru_cache(maxsize=1)
def example_config():
    """Load agent-rt.yaml from the current folder or ~/.agent-rt/config.yaml.

    See agent-rt.example.yaml at the repository root for a starting point.
    """
    return load_config()


def example_provider():
    return load_model(example_config())


def configured_model() -> str:
    model = example_config().default_model
    if not model:
        raise RuntimeError("Add a model to model_list in your agent-rt.yaml.")
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
