

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
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

class Incident:
    id: str = Field(..., description="Unique incident ID")
    type: IncidentType
    severity: SeverityLevel
    location: str
    people_affected: int
    resolved: bool = False


class Resource:
    type: str = Field(..., description="Type of resource (ambulance, firetruck, etc.)")
    available: int
    in_use: int


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

    amount: Optional[int] = Field(
        0, description="Amount of resource to allocate"
    )

    priority: Optional[int] = Field(
        None, description="Priority level for incident (1 = highest)"
    )

# -------------------------
# OBSERVATION MODEL
# -------------------------

class MyObservation(Observation):
    """
    Observation returned to the agent after each step.
    """

    # Core system state
    time_step: int = Field(..., description="Current time step")
    active_incidents: List[Incident] = Field(
        default_factory=list, description="List of active incidents"
    )

    resources: List[Resource] = Field(
        default_factory=list, description="Available resources"
    )

    # Aggregated metrics (important for learning signal)
    total_people_affected: int = Field(..., description="Total impacted population")
    resolved_incidents: int = Field(..., description="Number of resolved incidents")

    # System health indicators
    system_load: float = Field(
        ..., description="Ratio of used resources to total resources"
    )

    response_efficiency: float = Field(
        ..., description="Effectiveness of actions taken (0-1)"
    )

    # Episode signals
    done: bool = Field(..., description="Whether episode has ended")
    reward: float = Field(..., description="Reward from last step")
