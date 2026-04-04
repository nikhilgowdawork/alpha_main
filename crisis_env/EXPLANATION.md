# What I have Built

## Quick Answer: YES - The Agent Does 3 Tasks

You have **two different systems**:

---

## System 1: Crisis Environment (inference.py)
**Single long interaction** - Agent does ONE episode over multiple steps

- **What happens**: Agent takes actions in a crisis environment (dispatch teams, resolve incidents)
- **Rewards**: Gets partial rewards at each step based on performance
- **Goal**: Resolve all incidents to get `done=true`
- **Example run**:
  ```
  Step 1: [action] dispatch_team → [reward] -0.5 → [done] False
  Step 2: [action] allocate_resource → [reward] -0.3 → [done] False  
  Step 3: [action] resolve_incident → [reward] +10.0 → [done] True ✓
  ```
- **See it**: Run `python inference.py`

---

## System 2: Task Grader (baseline.py)
**Three separate isolated tasks** - Agent gets tested on 3 different problems

| Task | Type | What Agent Must Do | Score |
|------|------|-------------------|-------|
| **Easy** | Classification | Look at incident, classify urgency as "low"/"medium"/"high" | 0.0-1.0 |
| **Medium** | Allocation | Look at incident, list needed resources | 0.0-1.0 |
| **Hard** | Coordination | Coordinate response for multiple incidents with resource constraints | 0.0-1.0 |

- **How grading works**:
  - Easy: Exact match = 1.0, partially correct = 0.3, wrong = 0.0
  - Medium: Partial credit based on overlapping resources
  - Hard: Points for correct allocations, penalties for wrong ones

- **See it**: Run `python demo_tasks.py` (shows all 3 tasks working)

---

## Reward System - Is It Valid? ✓ YES

The reward combines **two types of signals**:

### 1. Step-by-step Rewards (Partial Progress)
```
Reward = -0.01 × people_affected 
       + 10.0 × incidents_resolved
       + 5.0 × high_severity_resolved
```

**Example**:
- If 50 people affected → -0.50 (penalty)
- When 1 incident resolves → +10.0 (bonus)
- When high-priority resolves → +5.0 (extra bonus)

**Why this is valid**:
- ✓ Encourages fast resolution (negative base reward incentivizes action)
- ✓ Rewards resolution (big +10 bonus)
- ✓ Prioritizes critical incidents (+5 bonus for high-severity)
- ✓ Shows partial progress (agent sees rewards improve as it acts)

### 2. Task Scores (0.0-1.0 Normalized)
Each task evaluates agent's quality:
- Easy task: Can you classify correctly? →  0.0-1.0
- Medium task: Do you pick right resources? → 0.0-1.0  
- Hard task: Can you coordinate complex scenarios? → 0.0-1.0

**Why this is valid**:
- ✓ All normalized to [0.0, 1.0] range (hackathon requirement)
- ✓ Supports partial credit (0.3, 0.5, 0.7 scores)
- ✓ Tests multiple difficulty levels
- ✓ Independent evaluation (each task grades separately)

---

## How To See Everything Working

### ✓ See the 3 Tasks:
```bash
python demo_tasks.py
```
Output shows:
- Task 1 score: 1.0 ✓
- Task 2 score: 0.7 ✓
- Task 3 score: 0.5 ✓

### ✓ See the Reward System:
```bash
python demo_rewards.py
```
Output shows step-by-step rewards and how they're calculated

### ✓ Check Files Exist:
```bash
ls models.py inference.py baseline.py tasks/task_*.py openenv.yaml
```

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `models.py` | Typed data models (MyAction, MyObservation) |
| `server/my_env_environment.py` | Crisis environment with reset/step/state |
| `tasks/task_easy.py` | Task 1 - urgency classification grader |
| `tasks/task_medium.py` | Task 2 - resource allocation grader |
| `tasks/task_hard.py` | Task 3 - multi-incident coordination grader |
| `inference.py` | Environment interaction script (structured logs) |
| `baseline.py` | Task grader runner (scores all 3 tasks) |
| `openenv.yaml` | OpenEnv spec configuration |
| `Dockerfile` | Container for deployment |
| `README.md` | Full documentation with HF Spaces support |

---