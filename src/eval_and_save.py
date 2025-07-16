import sys
import subprocess
from pathlib import Path

# Usage: python src/eval_and_save.py model_name [variant] [n_episodes] [steps_per_episode]
# Example: python src/eval_and_save.py gpt-4o base 5 3

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/eval_and_save.py model_name [variant] [n_episodes] [steps_per_episode]")
        sys.exit(1)
    model = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else "base"
    n_episodes = sys.argv[3] if len(sys.argv) > 3 else "5"
    steps_per_episode = sys.argv[4] if len(sys.argv) > 4 else "3"
    # Run eval.py
    cmd = [sys.executable, "src/eval.py", n_episodes, steps_per_episode, variant, model]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    # Save results to unique file
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    model_file = results_dir / f"eval_log_{model}.jsonl"
    main_file = results_dir / "eval_log.jsonl"
    main_file.replace(model_file)
    print(f"Saved results to {model_file}")

if __name__ == "__main__":
    main() 