


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

        '''put resources code here'''
        
    # -------------------------
    # STEP
    # -------------------------


        

    def step(self, action: MyAction) -> MyObservation:  # type: ignore[override]
    
        self._state.step_count += 1

        message = action.message
        length = len(message)

        # Simple reward: longer messages get higher rewards
        reward = length * 0.1

        return MyObservation(
            echoed_message=message,
            message_length=length,
            done=False,
            reward=reward,
            metadata={"original_message": message, "step": self._state.step_count},
        )

    @property
    def state(self) -> State:
      
        return self._state
