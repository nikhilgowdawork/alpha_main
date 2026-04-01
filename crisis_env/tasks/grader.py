from dataclasses import dataclass
from typing import Dict


@dataclass
class easytask:
    '''
    Easy task: Classify incident urgency  
    
    The agent classify the urgency level of given incident.
    Output should be one of :["low", "medium", "high"]

    '''

    incident : str
    expected_output: str

    def get_observation(self) -> Dict:
        """
        What the agent sees
        """
        return {
            "incident_description": self.incident,
            "task": "classify_urgency",
            "allowed_outputs": ["low", "medium", "high"]
        }

    def grade(self, predicted_output: str) -> float:
        """
        Deterministic grading
        """
        if not isinstance(predicted_output, str):
            return 0.0

        predicted_output = predicted_output.strip().lower()

        if predicted_output == self.expected_output:
            return 1.0

        # partial credit (slightly wrong but close)
        if predicted_output in ["low", "medium", "high"]:
            return 0.3

        return 0.0


def create_easy_task() -> easytask:
    """
    Factory method to generate a task instance
    """

    incident = "A person has minor cuts and is conscious after falling from a bicycle."
    expected_output = "low"

    return easytask(
        incident=incident,
        expected_output=expected_output
    )
