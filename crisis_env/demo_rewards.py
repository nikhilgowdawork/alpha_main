"""
DEMO: Shows the REWARD SYSTEM in the Crisis Environment
Demonstrates how rewards are calculated at each step
"""

from server.my_env_environment import MyEnvironment
from models import MyAction
import json

print("\n" + "-"*70)
print("CRISIS RESPONSE ENVIRONMENT - REWARD SYSTEM DEMO")
print("-"*70)

env = MyEnvironment()
obs = env.reset()

print(f"\n[RESET] Environment initialized")
print(f"  Initial observation:")
print(f"    - Time step: {obs.time_step}")
print(f"    - Active incidents: {len(obs.active_incidents)}")
print(f"    - People affected: {obs.total_people_affected}")
print(f"    - Initial reward: {obs.reward}")

print(f"\n" + "-"*70)
print("TAKING ACTIONS - REWARDS AT EACH STEP")
print("-"*70)

# Step 1: Dispatch a team (should reduce impact)
print(f"\n[STEP 1] Action: dispatch_team to incident 1")
action1 = MyAction(action_type="dispatch_team", incident_id="1")
obs1 = env.step(action1)
print(f"  Reward this step: {obs1.reward:.4f}")
print(f"  People affected: {obs1.total_people_affected}")
print(f"  Done? {obs1.done}")
print(f"  Calculation: {obs1.reward:.4f} (penalty for affected people)")

# Step 2: Allocate resources
print(f"\n[STEP 2] Action: allocate_resource (ambulance)")
action2 = MyAction(action_type="allocate_resource", resource_type="ambulance", amount=2)
obs2 = env.step(action2)
print(f"  Reward this step: {obs2.reward:.4f}")
print(f"  People affected: {obs2.total_people_affected}")
print(f"  Done? {obs2.done}")

# Step 3: Resolve an incident (bonus!)
print(f"\n[STEP 3] Action: resolve_incident (incident 1)")
action3 = MyAction(action_type="resolve_incident", incident_id="1")
obs3 = env.step(action3)
print(f"  Reward this step: {obs3.reward:.4f}")
print(f"  People affected: {obs3.total_people_affected}")
print(f"  Done? {obs3.done}")
print(f"  ✓ BONUS: +10 for resolving incident!")

print(f"\n" + "="*70)
print("REWARD CALCULATION FORMULA")
print("="*70)
print(f"""
Reward = -0.01 * (number of people affected by active incidents)
       + 10.0 * (number of incidents resolved this step)
       + 5.0 * (if high-severity incident resolved)
       - 0.1 * (penalties for inefficient actions)

Examples:
  - 5 people affected → -0.05
  - Resolve 1 incident → +10.0
  - Resolve high-severity → +5.0 (BONUS)
  - Do nothing → -penalty (encourages action)
""")

print(f"="*70 + "\n")
