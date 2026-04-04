from dataclasses import dataclass
from typing import List, Dict


@dataclass
class mediumtask:
    """
    Medium Task: Resource Allocation

    The agent must decide which emergency resources to deploy
    based on the incident description.

    Resources:
    - ambulance
    - fire_truck
    - police
    """"""
    Medium Task: Resource Allocation

    The agent must decide which emergency resources to deploy
    based on the incident description.

    Resources:
    - ambulance
    - fire_truck
    - police
    """

    incident : str
    expected_resources : list[str]

    def get_observation(self) -> Dict:

        #What the agent sees

        return {
            "incident_description": self.incident,
            "task": "allocate_resources",
            "available_resources": ["ambulance", "fire_truck", "police"]
        }
    
    def grade(self, predicted_resources: list[str]) -> float:

        #deterministic grading based on overlap

        if not isinstance(predicted_resources, list):
            return 0.0
        
        predicted = set([r.strip().lower() for r in predicted_resources])
        expected = set(self.expected_resources)

        if not predicted:
            return 0.0
        
        #correct selections
        correct = predicted & expected

        #penalty for extra unnecesary resources
        extra = predicted - expected

        score = 0.0

        #reward correct picks
        score += 0.7 * (len(correct) / len(expected))

        #penalize over-allocation
        if len(predicted) > 0:
            score -= 0.3 * (len(extra) / len(predicted))

        return max(0.0, min(1.0, score))


def create_medium_task() -> mediumtask:
    """
    Factory for medium task
    """

    incident = "A house is on fire and two people are injured inside."
    expected_resources = ["fire_truck", "ambulance"]

    return mediumtask(
        incident=incident,
        expected_resources=expected_resources
    )