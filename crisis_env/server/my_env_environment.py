from uuid import uuid4
from typing import List
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MyAction, MyObservation, Incident, Resource
except ImportError:
    from models import MyAction, MyObservation, Incident, Resource


class MyEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._time_step = 0

        self._incidents: List[Incident] = []
        self._resources: List[Resource] = []
        self._alert_broadcasted = False  # Track if alert was broadcasted this episode

    # -------------------------
    # RESET
    # -------------------------

    def reset(self) -> MyObservation:

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._time_step = 0
        self._alert_broadcasted = False

        self._incidents = [
            Incident(
                incident_id=f"inc_{i}",
                type=random.choice(["fire", "flood", "medical"]),
                severity=random.choice(["low", "medium", "high"]),
                location=f"zone_{i}",
                people_affected=random.randint(50, 200),
                resolved=False,
            )
            for i in range(2)
        ]

        self._resources = [
            Resource(type="ambulance", available=5, in_use=0),
            Resource(type="firetruck", available=3, in_use=0),
        ]

        return self._build_observation(reward=0.0, done=False)

    # -------------------------
    # STEP
    # -------------------------

    def step(self, action: MyAction):

        # Auto-reset if not reset yet
        if self._time_step == 0:
            self._state = State(episode_id=str(uuid4()), step_count=0)
            self._incidents = [
                Incident(
                    incident_id=f"inc_{i}",
                    type=random.choice(["fire", "flood", "medical"]),
                    severity=random.choice(["low", "medium", "high"]),
                    location=f"zone_{i}",
                    people_affected=random.randint(50, 200),
                    resolved=False,
                )
                for i in range(2)
            ]
            self._resources = [
                Resource(type="ambulance", available=5, in_use=0),
                Resource(type="firetruck", available=3, in_use=0),
            ]

        self._state.step_count += 1
        self._time_step += 1

        self._apply_action(action)
        self._update_dynamics()

        reward = self._compute_reward()
        done = self._is_done()

        return self._build_observation(reward=reward, done=done)

    # -------------------------
    # APPLY ACTION
    # -------------------------

    def _apply_action(self, action: MyAction):

        if action.action_type == "dispatch_team":
            # assign any available resource
            for res in self._resources:
                if res.available > 0:
                    res.available -= 1
                    res.in_use += 1
                    break

        elif action.action_type == "allocate_resource":
            # allocate specific resource if specified
            if action.resource_type:
                for res in self._resources:
                    if res.type == action.resource_type and res.available > 0:
                        res.available -= 1
                        res.in_use += 1
                        break
            else:
                # default to any available
                for res in self._resources:
                    if res.available > 0:
                        res.available -= 1
                        res.in_use += 1
                        break

        elif action.action_type == "request_backup":
            # request backup: add more resources
            for res in self._resources:
                res.available += 1

        elif action.action_type == "broadcast_alert":
            # broadcast alert: reduces people affected growth for next steps
            self._alert_broadcasted = True

        elif action.action_type == "prioritize_incident":
            # prioritize incident: mark as high priority (could affect resolution)
            if action.incident_id:
                for inc in self._incidents:
                    if inc.incident_id == action.incident_id:
                        inc.severity = "high"  # upgrade to high if not already

        elif action.action_type == "resolve_incident" and action.incident_id:
            for inc in self._incidents:
                if inc.incident_id == action.incident_id:
                    inc.resolved = True
                    # bonus for resolving high severity
                    if inc.severity == "high":
                        pass  # could add extra reward, but handled in _compute_reward

        # do_nothing does nothing

    # -------------------------
    # WORLD DYNAMICS
    # -------------------------

    def _update_dynamics(self):

        growth_factor = 0.5 if self._alert_broadcasted else 1.0
        self._alert_broadcasted = False  # reset after use

        for inc in self._incidents:
            if not inc.resolved:
                inc.people_affected += int(random.randint(5, 20) * growth_factor)

    # -------------------------
    # REWARD
    # -------------------------

    def _compute_reward(self) -> float:

        total_affected = sum(i.people_affected for i in self._incidents)
        resolved = sum(1 for i in self._incidents if i.resolved)
        high_severity_bonus = sum(5 for i in self._incidents if i.resolved and i.severity == "high")

        return resolved * 10.0 + high_severity_bonus - total_affected * 0.005

    # -------------------------
    # DONE
    # -------------------------

    def _is_done(self) -> bool:

        if not self._incidents:
            return False  # prevent empty list bug

        all_resolved = all(i.resolved for i in self._incidents)

        return all_resolved or self._time_step >= 50

    # -------------------------
    # BUILD OBSERVATION
    # -------------------------

    def _build_observation(self, reward: float, done: bool) -> MyObservation:

        total_affected = sum(i.people_affected for i in self._incidents)
        resolved_count = sum(1 for i in self._incidents if i.resolved)

        total_resources = sum(r.available + r.in_use for r in self._resources)
        used_resources = sum(r.in_use for r in self._resources)

        system_load = used_resources / total_resources if total_resources > 0 else 0
        efficiency = resolved_count / len(self._incidents) if self._incidents else 0

        return MyObservation(
            time_step=self._time_step,
            active_incidents=self._incidents,
            resources=self._resources,
            total_people_affected=total_affected,
            resolved_incidents=resolved_count,
            system_load=system_load,
            response_efficiency=efficiency,
            done=done,
            reward=reward,
        )

    # -------------------------
    # STATE META
    # -------------------------

    @property
    def state(self) -> State:
        return self._state