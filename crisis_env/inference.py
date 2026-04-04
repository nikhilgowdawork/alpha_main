"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import re
import textwrap
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from models import MyAction
from server.my_env_environment import MyEnvironment

load_dotenv()

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME", "crisis_response")
BENCHMARK = os.getenv("BENCHMARK", "crisis_response_env")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Max possible reward: each token contributes 0.1, across all steps
_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are interacting with a crisis response environment.
    Each turn you must choose exactly one valid environment action.
    Respond with only the action name, optionally followed by an incident ID for resolve_incident.
    Do not return JSON objects or extra explanation.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, observation: Dict, history: List[str]) -> str:
    incidents = observation.get("active_incidents", [])
    resources = observation.get("resources", [])

    incident_lines = []
    unresolved_ids = []
    for inc in incidents:
        incident_lines.append(
            f"- {inc['incident_id']}: {inc['type']} severity={inc['severity']} "
            f"people={inc['people_affected']} resolved={inc['resolved']}"
        )
        if not inc.get('resolved', False):
            unresolved_ids.append(inc['incident_id'])

    resource_lines = []
    for res in resources:
        resource_lines.append(
            f"- {res['type']}: available={res['available']} in_use={res['in_use']}"
        )

    history_lines = "\n".join(history[-4:]) if history else "None"

    resolve_options = "\n".join(f"resolve_incident {inc_id}" for inc_id in unresolved_ids) if unresolved_ids else "resolve_incident <incident_id> (no unresolved incidents)"

    incident_lines_str = '\n'.join(incident_lines) if incident_lines else '- none'
    resource_lines_str = '\n'.join(resource_lines) if resource_lines else '- none'

    return textwrap.dedent(
        f"""
        Step: {step}
        Time step: {observation.get('time_step')}
        Total affected: {observation.get('total_people_affected')}
        Resolved incidents: {observation.get('resolved_incidents')}
        System load: {observation.get('system_load')}
        Response efficiency: {observation.get('response_efficiency')}

        Incidents:
        {incident_lines_str}

        Resources:
        {resource_lines_str}

        Previous actions:
        {history_lines}

        Goal: Resolve all incidents as quickly as possible to maximize reward.
        CRITICAL: You MUST resolve incidents using 'resolve_incident <incident_id>' to stop the crisis and get positive rewards.
        Make reasonable decisions: dispatch resources to handle incidents, request backup when overloaded, broadcast alerts to slow spread, but PRIORITIZE RESOLVING INCIDENTS.

        Choose exactly one action:
        dispatch_team (send any available resource)
        allocate_resource <type> (allocate specific resource type)
        request_backup (call for more resources)
        broadcast_alert (alert public to reduce crisis spread)
        prioritize_incident <incident_id> (mark incident as high priority)
        {resolve_options}
        do_nothing
        """
    ).strip()


def parse_action_text(text: str) -> Optional[Dict]:
    if not text:
        return None

    normalized = text.strip().lower().split("\n")[0].strip()
    normalized = re.sub(r"[^a-z0-9_ ]+", "", normalized)
    parts = normalized.split()
    if not parts:
        return None

    action_type = parts[0]
    if action_type not in {
        "dispatch_team",
        "allocate_resource",
        "request_backup",
        "broadcast_alert",
        "prioritize_incident",
        "resolve_incident",
        "do_nothing",
    }:
        return None

    action = {"action_type": action_type}
    if action_type in ("allocate_resource", "prioritize_incident", "resolve_incident"):
        if len(parts) >= 2:
            if action_type == "allocate_resource":
                action["resource_type"] = parts[1]
            else:
                action["incident_id"] = parts[1]
        else:
            return None
    return action


def observation_to_dict(observation) -> Dict:
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if isinstance(observation, dict):
        return observation
    return getattr(observation, "__dict__", {})


def choose_fallback_action(observation: Dict) -> Dict:
    incidents = observation.get("active_incidents", [])
    resources = observation.get("resources", [])

    for res in resources:
        available = res.get("available") if isinstance(res, dict) else getattr(res, "available", 0)
        if available and available > 0:
            return {"action_type": "dispatch_team"}

    for inc in incidents:
        resolved = inc.get("resolved") if isinstance(inc, dict) else getattr(inc, "resolved", False)
        if not resolved:
            incident_id = inc.get("incident_id") if isinstance(inc, dict) else getattr(inc, "incident_id", None)
            return {"action_type": "resolve_incident", "incident_id": incident_id}

    return {"action_type": "do_nothing"}


def normalize_score(rewards: List[float]) -> float:
    if not rewards:
        return 0.0
    total = sum(rewards)
    normalized = (total + 50.0) / 100.0
    return max(0.0, min(1.0, normalized))


def create_action(payload: Dict) -> MyAction:
    return MyAction(
        action_type=payload["action_type"],
        incident_id=payload.get("incident_id"),
    )


async def main() -> None:
    if not API_KEY:
        raise ValueError("HF_TOKEN or API_KEY is required for inference.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = MyEnvironment()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset()
        observation = observation_to_dict(result)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            prompt = build_user_prompt(step, observation, history)
            model_output = None
            action_payload = None
            error_message = None

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                model_output = completion.choices[0].message.content.strip()
                action_payload = parse_action_text(model_output)
                if action_payload is None:
                    error_message = f"invalid output: {model_output!r}"
            except Exception as exc:
                error_message = str(exc)

            if not action_payload:
                action_payload = choose_fallback_action(observation)
                if model_output and not error_message:
                    error_message = f"fallback_used_from_model_output={model_output!r}"

            action = create_action(action_payload)
            result = env.step(action)
            observation = observation_to_dict(result)

            reward = result.reward if result.reward is not None else 0.0
            done = result.done
            action_str = action_payload["action_type"]
            if action_payload.get("incident_id"):
                action_str += f" {action_payload['incident_id']}"

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_message)
            history.append(f"{action_str} -> {reward:.2f}")

            if done:
                break

        score = normalize_score(rewards)
        success = bool(result.done)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
