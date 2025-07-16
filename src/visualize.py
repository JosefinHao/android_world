import json
from pathlib import Path
from collections import defaultdict, Counter

RESULTS_FILE = Path(__file__).parent.parent / "results" / "eval_log.jsonl"


def main():
    episodes_by_model = defaultdict(lambda: defaultdict(list))
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            model = entry.get("model", "unknown")
            episodes_by_model[model][entry["episode"]].append(entry)

    print("\n=== EVALUATION RESULTS BY MODEL ===\n")
    for model, episodes in episodes_by_model.items():
        print(f"=== Model: {model} ===")
        total_steps = 0
        correct_steps = 0
        hallucinations = 0
        episode_success = 0
        for epi, steps in episodes.items():
            all_correct = True
            for step in steps:
                total_steps += 1
                is_correct = step["correct"]
                if is_correct:
                    correct_steps += 1
                else:
                    all_correct = False
                if step["hallucination"]:
                    hallucinations += 1
                print(f"  Ep {epi} Step {step['step']}: LLM Action: {step['llm_action']}, GT: {step['ground_truth']}, Correct: {is_correct}, Hallucination: {step['hallucination']}")
            print(f"  Episode {epi} Success: {all_correct}\n")
            if all_correct:
                episode_success += 1
        n_episodes = len(episodes)
        print(f"Model: {model} | Episodes: {n_episodes} | Steps: {total_steps}")
        print(f"  Step accuracy: {correct_steps}/{total_steps} = {correct_steps/total_steps:.2f}")
        print(f"  Episode success rate: {episode_success}/{n_episodes} = {episode_success/n_episodes:.2f}")
        print(f"  Hallucinated actions: {hallucinations}/{total_steps} = {hallucinations/total_steps:.2f}")
        print("-----------------------------\n")

    # Overall summary
    print("=== OVERALL SUMMARY ===")
    all_steps = []
    for model, episodes in episodes_by_model.items():
        for epi, steps in episodes.items():
            all_steps.extend(steps)
    total_steps = len(all_steps)
    correct_steps = sum(1 for s in all_steps if s["correct"])
    hallucinations = sum(1 for s in all_steps if s["hallucination"])
    episode_success = sum(1 for (model, episodes) in episodes_by_model.items() for steps in episodes.values() if all(s["correct"] for s in steps))
    n_episodes = sum(len(episodes) for episodes in episodes_by_model.values())
    print(f"Total episodes: {n_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Step accuracy: {correct_steps}/{total_steps} = {correct_steps/total_steps:.2f}")
    print(f"Episode success rate: {episode_success}/{n_episodes} = {episode_success/n_episodes:.2f}")
    print(f"Hallucinated actions: {hallucinations}/{total_steps} = {hallucinations/total_steps:.2f}")

if __name__ == "__main__":
    main() 