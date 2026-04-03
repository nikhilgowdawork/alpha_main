from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field
from typing import List, Literal, Optional


# -------------------------
# ENUM-LIKE TYPES
# -------------------------

ActionType = Literal[
    "dispatch_team",
    "allocate_resource",
    "request_backup",
    "broadcast_alert",
    "prioritize_incident",
    "resolve_incident",
    "do_nothing"
]

IncidentType = Literal[
    "fire",
    "flood",
    "earthquake",
    "cyberattack",
    "medical"
]

SeverityLevel = Literal["low", "medium", "high", "critical"]


# -------------------------
# NESTED STRUCTURES
# -------------------------

class Incident(BaseModel):
    incident_id: str = Field(..., description="Unique incident ID")
    type: IncidentType
    severity: SeverityLevel
    location: str
    people_affected: int = Field(..., ge=0)
    resolved: bool = Field(default=False)


class Resource(BaseModel):
    type: str = Field(..., description="Type of resource (ambulance, fire_truck, etc.)")
    available: int = Field(..., ge=0)
    in_use: int = Field(..., ge=0)


# -------------------------
# ACTION MODEL
# -------------------------

class MyAction(Action):

    action_type: ActionType = Field(..., description="Type of action to execute")

    incident_id: Optional[str] = Field(
        None, description="Target incident ID (if applicable)"
    )

    resource_type: Optional[str] = Field(
        None, description="Type of resource to allocate"
    )

    amount: int = Field(
        default=0, ge=0, description="Amount of resource to allocate"
    )

    priority: Optional[int] = Field(
        None, ge=1, le=5, description="Priority level (1 = highest)"
    )


# -------------------------
# OBSERVATION MODEL
# -------------------------

class MyObservation(Observation):
    """
    Observation returned to the agent after each step.
    """

    # Time tracking
    time_step: int = Field(..., ge=0, description="Current time step")

    # Core state
    active_incidents: List[Incident] = Field(
        default_factory=list,
        description="List of active (unresolved) incidents"
    )

    resources: List[Resource] = Field(
        default_factory=list,
        description="Current resource availability"
    )

    # Metrics
    total_people_affected: int = Field(..., ge=0)
    resolved_incidents: int = Field(..., ge=0)

    # System health
    system_load: float = Field(
        ..., ge=0.0, le=1.0,
        description="Used resources / total resources"
    )

    response_efficiency: float = Field(
        ..., ge=0.0, le=1.0,
        description="Quality of decisions (computed internally)"
    )

    # OpenEnv required signals
    done: bool = Field(..., description="Episode termination flag")
    reward: float = Field(..., description="Reward from last action")