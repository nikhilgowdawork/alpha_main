from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, ConfigDict
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
    model_config = ConfigDict(from_attributes=True)

    incident_id: str = Field(..., description="Unique incident ID")
    type: IncidentType
    severity: SeverityLevel
    location: str
    people_affected: int = Field(..., ge=0)
    resolved: bool = Field(default=False)


class Resource(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    type: str = Field(..., description="Resource type (ambulance, firetruck, etc.)")
    available: int = Field(..., ge=0)
    in_use: int = Field(..., ge=0)


# -------------------------
# ACTION MODEL
# -------------------------

class MyAction(Action):

    action_type: ActionType = Field(
        ..., 
        description="Type of action",
        examples=["dispatch_team"]
    )

    incident_id: Optional[str] = Field(
        None,
        description="Target incident ID (required for resolve actions)",
        examples=["inc_0"]
    )

    resource_type: Optional[str] = Field(
        None,
        description="Type of resource to allocate"
    )

    amount: int = Field(
        default=0,
        ge=0,
        description="Amount of resource to allocate"
    )

    priority: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Priority level (1 = highest)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action_type": "dispatch_team",
                "incident_id": None,
                "resource_type": None,
                "amount": 0,
                "priority": None
            }
        }
    )

# Rebuild the model to apply the config
MyAction.model_rebuild()


# -------------------------
# OBSERVATION MODEL
# -------------------------

class MyObservation(Observation):

    model_config = ConfigDict(from_attributes=True)

    # Time
    time_step: int = Field(..., ge=0)

    # Core state
    active_incidents: List[Incident] = Field(default_factory=list)
    resources: List[Resource] = Field(default_factory=list)

    # Metrics
    total_people_affected: int = Field(..., ge=0)
    resolved_incidents: int = Field(..., ge=0)

    # System health
    system_load: float = Field(..., ge=0.0, le=1.0)
    response_efficiency: float = Field(..., ge=0.0, le=1.0)

    # Required signals
    done: bool
    reward: float