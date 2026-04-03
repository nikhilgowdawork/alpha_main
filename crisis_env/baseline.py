import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from tasks.task_easy import create_easy_task
from tasks.task_medium import create_medium_task
from tasks.task_hard import create_hard_task

#load env variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

#initialize HF client

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

def run_task(task):
    observation = task.get_observation()

    prompt = f"""
You are an expert crisis response coordinator.

Given the situation below, make the best decision.

IMPORTANT:
- Return ONLY the answer
- No explanations
- Follow expected format strictly

Situation:
{observation}
"""
    response = client.text_generation(
        prompt,
        max_new_tokens=200,
        temperature=0

    )

    output = response.strip()

    
    #try prasing JSON (for  medium/hard tasks)

    try:
        parsed_outout = json.loads(output)

    except Exception:
        paersed_output = output

        score = task.grade(parsed_outout)

        return output, score
    

def  main():
    tasks = [
        ("easy",create_easy_task()),
        ("medium", create_medium_task()),
        ("hard", create_hard_task())
    ]

    total_score = 0

    for name, task in tasks:
        print(f"\n=== {name} Task ===")
        
        output, score = run_task(task)
        
        print("Model Output:", output)
        print("Score:", score)
        
        total_score += score
        
    final_score = total_score / len(tasks)
        
    print("\n=== FINAL SCORE ===")
    print(final_score)
        
if __name__ == "__main__":
    main()
    

    
    
