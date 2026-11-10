"""MCP capability filtering, access policy, and the authorized MCP client."""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ext.runtime.authorization.grants import CapabilityGrant
from ext.runtime.authorization.identity import (
    AuthenticationContext,
    AuthorizationEngine,
    AuthorizationRequirement,
)
from ext.runtime.authorization.policy import PolicyEngine, PolicyRequest
from ext.runtime.protocols.mcp import (
    MCPCapabilityKind,
    MCPClient,
    MCPPrompt,
    MCPResource,
    MCPTool,
)


@dataclass(frozen=True)
class MCPCapabilityFilter:
    tools: tuple[str, ...] = ("*",)
    resources: tuple[str, ...] = ("*",)
    prompts: tuple[str, ...] = ("*",)

    def allows(self, kind: MCPCapabilityKind, name: str) -> bool:
        patterns = {
            "tool": self.tools,
            "resource": self.resources,
            "prompt": self.prompts,
        }[kind]
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


@dataclass(frozen=True)
class MCPAccessPolicy:
    authentication: AuthenticationContext | None = None
    authorization_engine: AuthorizationEngine = field(
        default_factory=AuthorizationEngine
    )
    requirements: Mapping[str, AuthorizationRequirement] = field(default_factory=dict)
    policy_engine: PolicyEngine | None = None
    capability_filter: MCPCapabilityFilter = field(default_factory=MCPCapabilityFilter)
    capability_grant: CapabilityGrant | None = None

    def check(
        self,
        *,
        server: str,
        kind: MCPCapabilityKind,
        name: str,
        action: str,
    ) -> None:
        if not self.capability_filter.allows(kind, name):
            raise PermissionError(f"MCP {kind} is not approved: {name}")
        if (
            kind == "tool"
            and self.capability_grant is not None
            and self.capability_grant.tools
            and not self.capability_grant.allows_tool(name)
        ):
            raise PermissionError(f"MCP tool is outside the capability grant: {name}")
        if self.authentication is None and self.requirements:
            # Configured requirements cannot be evaluated without an identity;
            # silently skipping them would turn a misconfiguration into access.
            raise PermissionError("MCP policy requires an authenticated principal")
        if self.authentication is not None:
            requirement = self.requirements.get(
                f"{kind}:{action}",
                self.requirements.get(kind, AuthorizationRequirement()),
            )
            self.authorization_engine.check(self.authentication, requirement)
            credential = self.authentication.credential
            if (
                credential is not None
                and self.capability_grant is not None
                and not self.capability_grant.allows_credential(credential.id)
            ):
                raise PermissionError("MCP credential is outside the capability grant")
        if self.policy_engine is not None:
            subject = (
                self.authentication.principal.id
                if self.authentication is not None
                else None
            )
            self.policy_engine.check(
                PolicyRequest(
                    domain="mcp",
                    action=f"{kind}:{action}",
                    subject=subject,
                    resource=f"{server}:{name}",
                    attributes={"server": server, "kind": kind},
                )
            )


class AuthorizedMCPClient:
    def __init__(self, client: MCPClient, policy: MCPAccessPolicy) -> None:
        self.client = client
        self.policy = policy

    def _server_name(self) -> str:
        return self.client._ensure_initialized().name

    async def list_tools(self) -> tuple[MCPTool, ...]:
        server = self._server_name()
        tools = await self.client.list_tools()
        return tuple(
            tool
            for tool in tools
            if self._allowed(server, "tool", tool.name, "discover")
        )

    async def list_resources(self) -> tuple[MCPResource, ...]:
        server = self._server_name()
        resources = await self.client.list_resources()
        return tuple(
            resource
            for resource in resources
            if self._allowed(server, "resource", resource.uri, "discover")
        )

    async def list_prompts(self) -> tuple[MCPPrompt, ...]:
        server = self._server_name()
        prompts = await self.client.list_prompts()
        return tuple(
            prompt
            for prompt in prompts
            if self._allowed(server, "prompt", prompt.name, "discover")
        )

    def _allowed(
        self,
        server: str,
        kind: MCPCapabilityKind,
        name: str,
        action: str,
    ) -> bool:
        try:
            self.policy.check(
                server=server,
                kind=kind,
                name=name,
                action=action,
            )
        except PermissionError:
            return False
        return True

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.policy.check(
            server=self._server_name(),
            kind="tool",
            name=name,
            action="invoke",
        )
        return await self.client.call_tool(name, arguments)

    async def read_resource(self, uri: str) -> Mapping[str, Any]:
        self.policy.check(
            server=self._server_name(),
            kind="resource",
            name=uri,
            action="read",
        )
        return await self.client.read_resource(uri)

    async def get_prompt(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.policy.check(
            server=self._server_name(),
            kind="prompt",
            name=name,
            action="get",
        )
        return await self.client.get_prompt(name, arguments)
