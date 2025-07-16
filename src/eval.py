import sys
import os
import json
import re
from pathlib import Path
from episode_loader import generate_multi_step_episodes
from agent import format_prompt, call_openai, call_openai_function_calling, call_claude, TEMPLATES
from dotenv import load_dotenv
load_dotenv()

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_FILE = RESULTS_DIR / "eval_log.jsonl"

def extract_ui_elements(observation):
    match = re.search(r'UI Elements: \[(.*?)\]', observation)
    if match:
        elems = match.group(1)
        return [e.strip().strip('"') for e in elems.split(',')]
    return re.findall(r'"([^"]+)"', observation)

def hallucination_check(llm_action, ui_elements):
    match = re.match(r'CLICK\(["\'](.+?)["\']\)', llm_action or "")
    if match:
        elem = match.group(1)
        return elem in ui_elements, elem
    return False, None

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
    steps_per_episode = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    variant = sys.argv[3] if len(sys.argv) > 3 else "base"
    model = sys.argv[4] if len(sys.argv) > 4 else "gpt-4o"
    if variant not in TEMPLATES:
        print(f"Unknown prompt variant '{variant}'. Using 'base'.")
        variant = "base"
    print(f"[Prompt variant: {variant}] [Episodes: {n}] [Steps per episode: {steps_per_episode}] [Model: {model}]")
    episodes = generate_multi_step_episodes(n, steps_per_episode)
    total_steps = 0
    correct_steps = 0
    episode_success = 0
    hallucinations = 0
    with open(LOG_FILE, "w", encoding="utf-8") as logf:
        for epi, episode in enumerate(episodes):
            print(f"\n=== EPISODE {epi+1} ===")
            print("Goal:", episode["goal"])
            history = []
            all_correct = True
            for step_idx, (obs, gt_action) in enumerate(episode["steps"]):
                prompt = format_prompt(episode["goal"], obs, variant, history)
                if model.startswith("claude"):
                    llm_action = call_claude(prompt, model=model)
                elif variant == "function_calling":
                    llm_action = call_openai_function_calling(prompt, model=model)
                else:
                    llm_action = call_openai(prompt, model=model)
                is_correct = (llm_action.strip() == gt_action.strip()) if llm_action else False
                ui_elements = extract_ui_elements(obs)
                is_valid, clicked_elem = hallucination_check(llm_action, ui_elements)
                if not is_valid:
                    print(f"[HALLUCINATION] Step {step_idx+1}: LLM clicked '{clicked_elem}' not in UI elements: {ui_elements}")
                    hallucinations += 1
                else:
                    print(f"Step {step_idx+1}: LLM clicked valid UI element: '{clicked_elem}'")
                print(f"Step {step_idx+1} LLM Action: {llm_action}")
                print(f"Step {step_idx+1} Ground Truth: {gt_action}")
                print(f"Step {step_idx+1} Correct: {is_correct}")
                logf.write(json.dumps({
                    "episode": epi+1,
                    "step": step_idx+1,
                    "goal": episode["goal"],
                    "observation": obs,
                    "llm_action": llm_action,
                    "ground_truth": gt_action,
                    "correct": is_correct,
                    "hallucination": not is_valid,
                    "clicked_elem": clicked_elem,
                    "ui_elements": ui_elements,
                    "history": history.copy(),
                    "model": model
                }) + "\n")
                total_steps += 1
                if is_correct:
                    correct_steps += 1
                else:
                    all_correct = False
                history.append((obs, llm_action))
            if all_correct:
                episode_success += 1
            print(f"Episode {epi+1} success: {all_correct}")
    print(f"\nStep accuracy: {correct_steps}/{total_steps} = {correct_steps/total_steps:.2f}")
    print(f"Episode success rate: {episode_success}/{n} = {episode_success/n:.2f}")
    print(f"Hallucinated actions: {hallucinations}/{total_steps} = {hallucinations/total_steps:.2f}")
    print(f"Log written to {LOG_FILE}")

if __name__ == "__main__":
    main() 