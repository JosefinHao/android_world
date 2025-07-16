import subprocess
import sys
from pathlib import Path

MODELS = [
    "gpt-4o",
    "gpt-3.5-turbo",
    "claude-3-opus-20240229",
    # Add more models as needed
]
VARIANT = "base"  # or "few_shot", "function_calling"
N_EPISODES = "10"  # Changed from 5 to 10
STEPS_PER_EPISODE = "3"


def run_eval_and_save(model, variant=VARIANT, n_episodes=N_EPISODES, steps_per_episode=STEPS_PER_EPISODE):
    cmd = [
        sys.executable, "src/eval_and_save.py",
        model, variant, n_episodes, steps_per_episode
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def merge_results(model_files):
    cmd = [sys.executable, "src/merge_results.py"] + model_files
    print(f"Merging: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    results_dir = Path("results")
    model_files = []
    for model in MODELS:
        run_eval_and_save(model)
        model_file = results_dir / f"eval_log_{model}.jsonl"
        model_files.append(str(model_file))
    merge_results(model_files)
    print("Launching Streamlit...")
    subprocess.run(["streamlit", "run", "src/streamlit_app.py"])

if __name__ == "__main__":
    main() 