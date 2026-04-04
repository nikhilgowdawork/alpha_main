from dataclasses import dataclass
from typing import List, Dict


@dataclass
class hardtask:
    """
    Hard Task: Multi-Incident Crisis Coordination

    Agent must:
    - prioritize incidents
    - allocate limited resources
    - make trade-offs

    Output format expected:
    {
        "plan": [
            {"incident_id": int, "resources": [str]}
        ]
    }
    """
    incidents : List[Dict]
    available_resources : List[str]
    expected_plan : List[Dict]

    def get_observation(self) -> Dict:

        # what agent sees

        return {
            "task" : "multi_incident_response",
            "incidents" : self.incidents,
            "available_resources" : self.available_resources,
            "instructions" : "prioritize high severity incidents and allocate resources efficiently."
        }
    
    def grade(self,predicted_plan: Dict) -> float:


        if not isinstance(predicted_plan, dict):
            return 0.0
        
        if "plan" not in predicted_plan:
            return 0.0
        
        predicted = predicted_plan["plan"]

        if not isinstance(predicted, list): 
            return 0.0
        
        score = 0.0

        # Convert the expected into Dict for easy lookup --- 
        expected_map = {item["incident_id"]: set(item["resources"]) for item in self.expected_plan}

        total_incidents = len(expected_map)

        if total_incidents == 0:
            return 0.0
        
        used_resources = []

        for item in predicted:
            if not isinstance(item, dict):
                continue

            incident_id = item.get("incident_id")
            resources = item.get("resources", [])

            if incident_id not in expected_map:
                continue

            predicted_set = set([ r.strip().lower() for r in resources])
            expected_set = expected_map[incident_id]

            correct = predicted_set & expected_set   #intersection
            extra = predicted_set - expected_set  #subtraction


            #reward coorect allocation per incident
            score += 0.5 * (len(correct) / len(expected_set))

            #penalize wrong allocation
            if len(predicted_set) > 0:
                score -= 0.2 * (len(extra) / len(predicted_set))

            used_resources.extend(list(predicted_set))

        
         # --- global penalty: resource overuse ---
        if len(used_resources) > len(self.available_resources):
            score -= 0.3

        # normalize
        score = score / total_incidents

        return max(0.0, min(1.0, score))


def create_hard_task() -> hardtask:
    """
    Factory for hard task
    """

    incidents = [
        {"incident_id": 1, "type": "fire", "severity": "high"},
        {"incident_id": 2, "type": "accident", "severity": "medium"}
    ]

    available_resources = ["fire_truck", "ambulance"]

    expected_plan = [
        {"incident_id": 1, "resources": ["fire_truck"]},
        {"incident_id": 2, "resources": ["ambulance"]}
    ]

    return hardtask(
        incidents=incidents,
        available_resources=available_resources,
        expected_plan=expected_plan
    )



