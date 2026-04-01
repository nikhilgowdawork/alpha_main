


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
        """Initialize the my_env environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        #internal world state 
        self._incidents: list[Incident] = []
        self._resources: list[Resource] = []
        self._time_step = 0

    # -------------------------
    # RESET
    # -------------------------

    def reset(self) -> MyObservation:
        
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._time_step = 0

        #initialize incidents
        self._incidents = [
            Incident(
                id=f"inc_{i}",
                type=random.choice(["fire", "flood", "medical"]),
                severity=random.choice(["low", "medium", "high"]),
                location=f"zone_{i}",
                people_affected=random.randint(50, 200),
                resolved=False,
            )
            for i in range(2)
        ]

        # initialize resources
        self._resources = [
            Resource(type="ambulance", available=5, in_use=0),
            Resource(type="firetruck", available=3, in_use=0),
        ]

        return self._build_observation(reward=0.0, done=False)
        
    # -------------------------
    # STEP
    # -------------------------
  

    def step(self, action: MyAction) -> MyObservation :

        self._state_meta.step_count += 1
        self._time_step += 1

        # 1. apply action
        self._apply_action(action)

        # 2. update world
        self._update_dynamics()

        # 3. compute reward
        reward = self._compute_reward()

        # 4. check done
        done = self._is_done()

        return self._build_observation(reward=reward, done=done)
    

    # -------------------------
    # APPLY ACTION
    # -------------------------

    def _apply_action(self, action: MyAction):

            if action.action_type == "dispatch_team" and action.incident_id:
                for res in self._resources:
                    if res.available > 0:
                        res.available -= 1
                        res.in_use += 1
                        break

            elif action.action_type == "resolve_incident" and action.incident_id:
                for inc in self._incidents:
                    if inc.id == action.incident_id:
                        inc.resolved = True



    # -------------------------
    # WORLD DYNAMICS
    # -------------------------

    def _update_dynamics(self):

        for inc in self._incidents:
            if not inc.resolved:
                # situation worsens
                inc.people_affected += random.randint(5, 20)



    # -------------------------
    # REWARD
    # -------------------------

    def _compute_reward(self) -> float:

        total_affected = sum(i.people_affected for i in self._incidents)
        resolved = sum(1 for i in self._incidents if i.resolved)

        # reward logic
        reward = (
            resolved * 10.0
            - total_affected * 0.01
        )

        return reward
    
    
    # -------------------------
    # DONE
    # -------------------------

    def _is_done(self) -> bool:
    
            all_resolved = all(i.resolved for i in self._incidents)
    
            return (
                all_resolved
                or self._time_step >= 50
            )
        
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
            return self._state_meta
    