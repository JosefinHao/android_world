import sys
import json
from pathlib import Path

# Usage: python src/merge_results.py results/eval_log_gpt4o.jsonl results/eval_log_claude.jsonl ...
# Output: results/eval_log.jsonl

def main():
    if len(sys.argv) < 3:
        print("Usage: python src/merge_results.py file1.jsonl file2.jsonl ...")
        sys.exit(1)
    input_files = sys.argv[1:]
    merged = {}
    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                key = (entry.get("model", "unknown"), entry["episode"], entry["step"])
                merged[key] = entry  # Last one wins if duplicate
    # Write merged results
    output_file = Path(input_files[0]).parent / "eval_log.jsonl"
    with open(output_file, "w", encoding="utf-8") as out:
        for entry in merged.values():
            out.write(json.dumps(entry) + "\n")
    print(f"Merged {len(input_files)} files into {output_file} ({len(merged)} unique entries)")

if __name__ == "__main__":
    main() 