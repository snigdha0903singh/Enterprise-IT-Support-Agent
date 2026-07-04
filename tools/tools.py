from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    arguments: dict[str, str]
    function: Callable[..., Any]


def create_access_request(system: str, employee: str) -> str:
    return f"Access request submitted for {employee} to {system}."


def reset_password(employee: str) -> str:
    return f"Password reset initiated for {employee}."


def open_incident(system: str, severity: str) -> str:
    return f"{severity} incident opened for {system}."


def assign_project_owner(project: str, owner: str) -> str:
    return f"{owner} assigned as owner for {project}."


def find_approver(policy: str) -> str:
    return f"Approver lookup started for policy {policy}."


def generate_change_request(system: str) -> str:
    return f"Change request generated for {system}."


def schedule_maintenance(system: str) -> str:
    return f"Maintenance scheduling started for {system}."


TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "create_access_request": ToolDefinition(
        name="create_access_request",
        description="Submit an access request when a user asks for an employee to get access to a system, application, or reporting tool.",
        arguments={
            "system": "The target system, application, platform, or reporting tool that access is needed for.",
            "employee": "The employee who needs access.",
        },
        function=create_access_request,
    ),
    "reset_password": ToolDefinition(
        name="reset_password",
        description="Start a password reset for an employee.",
        arguments={
            "employee": "The employee whose password should be reset.",
        },
        function=reset_password,
    ),
    "open_incident": ToolDefinition(
        name="open_incident",
        description="Open an incident when something is broken, degraded, unavailable, slow, or causing business impact.",
        arguments={
            "system": "The affected system or service.",
            "severity": "The incident severity, such as low, medium, high, critical, Sev1, Sev2, Sev3, or Sev4.",
        },
        function=open_incident,
    ),
    "assign_project_owner": ToolDefinition(
        name="assign_project_owner",
        description="Assign or change the owner for a project.",
        arguments={
            "project": "The project that needs an owner.",
            "owner": "The employee who should own the project.",
        },
        function=assign_project_owner,
    ),
    "find_approver": ToolDefinition(
        name="find_approver",
        description="Find the approver required by a named access policy, security policy, or approval policy.",
        arguments={
            "policy": "The policy whose approver should be found.",
        },
        function=find_approver,
    ),
    "generate_change_request": ToolDefinition(
        name="generate_change_request",
        description="Generate a change request for planned modifications to a system.",
        arguments={
            "system": "The system that will be changed.",
        },
        function=generate_change_request,
    ),
    "schedule_maintenance": ToolDefinition(
        name="schedule_maintenance",
        description="Schedule a maintenance window for a system.",
        arguments={
            "system": "The system that needs maintenance.",
        },
        function=schedule_maintenance,
    ),
}


TOOLS: dict[str, Callable[..., Any]] = {
    name: definition.function for name, definition in TOOL_DEFINITIONS.items()
}
