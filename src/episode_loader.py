import re
import random
from pathlib import Path

TASKS_FILE = Path(__file__).parent.parent / "android_world" / "task_evals" / "information_retrieval" / "proto" / "tasks.textproto"


def parse_first_task():
    """Parse the first tasks { ... } block and fill in parameters with sample values."""
    return parse_n_tasks(1)[0]


def parse_n_tasks(n=3):
    """Parse the first n tasks { ... } blocks and fill in parameters with sample values."""
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Find all tasks { ... } blocks
    blocks = re.findall(r"tasks \{(.*?)\n\}", text, re.DOTALL)
    episodes = []
    for block in blocks[:n]:
        # Extract prompt
        prompt_match = re.search(r'prompt: "(.*?)"', block)
        prompt = prompt_match.group(1) if prompt_match else ""

        # Extract relevant_state (just the first line for demo)
        state_match = re.search(r'relevant_state \{(.*?)success_criteria', block, re.DOTALL)
        relevant_state = state_match.group(1).strip() if state_match else ""

        # Extract task_params
        param_blocks = re.findall(r'task_params \{(.*?)\}', block, re.DOTALL)
        params = {}
        for pb in param_blocks:
            name_match = re.search(r'name: "(.*?)"', pb)
            name = name_match.group(1) if name_match else None
            values = re.findall(r'possible_values: "(.*?)"', pb)
            if name and values:
                params[name] = values

        # Fill in parameters with a random value (or first value)
        filled = {k: random.choice(v) for k, v in params.items()}
        goal = prompt
        for k, v in filled.items():
            goal = goal.replace(f"{{{k}}}", v)
        # For observation, just show the relevant_state with params filled
        observation = relevant_state
        for k, v in filled.items():
            observation = observation.replace(f"{{{k}}}", v)

        episodes.append({
            "goal": goal,
            "observation": observation,
            "params": filled,
        })
    return episodes


def generate_multi_step_episodes(n=3, steps_per_episode=3):
    """Generate mock multi-step episodes. Each episode is a dict with goal and steps: [(obs, action), ...]"""
    base_episodes = parse_n_tasks(n)
    multi_episodes = []
    for ep in base_episodes:
        steps = []
        for i in range(steps_per_episode):
            # Vary the observation and action for each step
            obs = ep["observation"] + f"\nUI Elements: [\"Apps\", \"Data Dive\", \"Settings\", \"Step{i+1}\"]"
            if i < steps_per_episode - 1:
                action = f'CLICK("Apps")'
            else:
                action = f'CLICK("Data Dive")'  # Final step is the goal
            steps.append((obs, action))
        multi_episodes.append({
            "goal": ep["goal"],
            "steps": steps
        })
    return multi_episodes

if __name__ == "__main__":
    episodes = generate_multi_step_episodes(2, 3)
    for i, episode in enumerate(episodes):
        print(f"--- MULTI-STEP EPISODE {i+1} ---")
        print("Goal:", episode["goal"])
        for j, (obs, act) in enumerate(episode["steps"]):
            print(f"Step {j+1} Observation:", obs)
            print(f"Step {j+1} Ground Truth Action:", act) 