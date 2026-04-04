"""
DEMO: Shows the 3 tasks and their scoring system
No API keys required - uses mock responses
"""

from tasks.task_easy import create_easy_task
from tasks.task_medium import create_medium_task
from tasks.task_hard import create_hard_task

print("\n" + "-"*60)
print("CRISIS RESPONSE - 3 TASKS DEMO")
print("-"*60)

# ============================================================
# TASK 1: EASY - Urgency Classification
# ============================================================
print("\n[TASK 1] EASY - Urgency Classification")
print("-" * 60)
task_easy = create_easy_task()
obs_easy = task_easy.get_observation()
print(f"Observation (what agent sees):")
print(f"  {obs_easy['task']}")
print(f"\nExpected answer: 'low'")
print(f"\nAgent's answer: 'low' (CORRECT)")
score_easy = task_easy.grade('low')
print(f"Score: {score_easy} ✓")
print(f"Valid range: [0.0, 1.0]")

# ============================================================
# TASK 2: MEDIUM - Resource Allocation
# ============================================================
print("\n[TASK 2] MEDIUM - Resource Allocation")
print("-" * 60)
task_medium = create_medium_task()
obs_medium = task_medium.get_observation()
print(f"Observation (what agent sees):")
print(f"  {obs_medium['task']}")
print(f"\nExpected answer: ['fire_truck', 'ambulance']")
print(f"\nAgent's answer: ['fire_truck', 'ambulance'] (CORRECT)")
score_medium = task_medium.grade(['fire_truck', 'ambulance'])
print(f"Score: {score_medium} ✓")
print(f"Valid range: [0.0, 1.0]")

# ============================================================
# TASK 3: HARD - Multi-Incident Coordination
# ============================================================
print("\n[TASK 3] HARD - Multi-Incident Coordination")
print("-" * 60)
task_hard = create_hard_task()
obs_hard = task_hard.get_observation()
print(f"Observation (what agent sees):")
print(f"  {obs_hard['task']}")
print(f"\nExpected answer: Allocation plan with correct resources for 2 incidents")
agent_plan = {'plan': [{'incident_id': 1, 'resources': ['fire_truck']}, {'incident_id': 2, 'resources': ['ambulance']}]}
print(f"\nAgent's answer: {agent_plan}")
score_hard = task_hard.grade(agent_plan)
print(f"Score: {score_hard:.2f} ✓")
print(f"Valid range: [0.0, 1.0]")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "-"*60)
print("SUMMARY: 3 GRADED TASKS")
print("-"*60)
print(f"✓ Task 1 (Easy):   Score = {score_easy} (Urgency classification)")
print(f"✓ Task 2 (Medium): Score = {score_medium:.2f} (Resource allocation)")
print(f"✓ Task 3 (Hard):   Score = {score_hard:.2f} (Multi-incident coordination)")
print(f"\nAll scores are in valid range [0.0, 1.0]")
print(f"Each task can award partial credit (e.g., 0.5, 0.8)")
print("-"*60 + "\n")
