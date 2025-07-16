from episode_loader import parse_first_task
from pathlib import Path
import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()

PROMPT_DIR = Path(__file__).parent.parent / "prompts"
TEMPLATES = {
    "base": PROMPT_DIR / "base_template.txt",
    "few_shot": PROMPT_DIR / "few_shot_template.txt",
    "self_reflection": PROMPT_DIR / "self_reflection_template.txt",
    "function_calling": PROMPT_DIR / "base_template.txt",  # fallback, not used
}

# --- OpenAI integration ---
try:
    import openai
except ImportError:
    openai = None

def load_prompt_template(variant="base"):
    path = TEMPLATES.get(variant, TEMPLATES["base"])
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_history(history):
    if not history:
        return ""
    lines = []
    for i, (obs, act) in enumerate(history):
        lines.append(f"Step {i+1} Observation:\n{obs}")
        lines.append(f"Step {i+1} Action: {act}")
    return "\n".join(lines) + "\n"

def format_prompt(goal, observation, variant="base", history=None):
    if variant == "function_calling":
        # For function-calling, just use a simple instruction
        return f"Goal: {goal}\nObservation:\n{observation}\nWhat is the next best action?"
    template = load_prompt_template(variant)
    history_str = format_history(history) if history else ""
    return template.format(goal=goal, observation=observation, history=history_str)

def call_openai(prompt, model="gpt-3.5-turbo"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not openai:
        print("[ERROR] openai package not installed. Run: pip install openai")
        return None
    if not api_key:
        print("[ERROR] OPENAI_API_KEY environment variable not set.")
        return None
    try:
        # For openai>=1.0.0
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        return None

def call_openai_function_calling(prompt, model="gpt-4o"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not openai:
        print("[ERROR] openai package not installed. Run: pip install openai")
        return None
    if not api_key:
        print("[ERROR] OPENAI_API_KEY environment variable not set.")
        return None
    functions = [
        {
            "name": "click",
            "description": "Click a UI element by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {"type": "string", "description": "The UI element to click."}
                },
                "required": ["element"]
            }
        }
    ]
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            functions=functions,
            function_call={"name": "click"},
            temperature=0.2,
            max_tokens=256,
        )
        msg = response.choices[0].message
        if msg.function_call and msg.function_call.name == "click":
            args = json.loads(msg.function_call.arguments)
            return f'CLICK("{args["element"]}")'
        return None
    except Exception as e:
        print(f"[ERROR] OpenAI function-calling failed: {e}")
        return None

# --- Claude (Anthropic) integration ---
def call_claude(prompt, model="claude-3-opus-20240229"):
    try:
        import anthropic
    except ImportError:
        print("[ERROR] anthropic package not installed. Run: pip install anthropic")
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[ERROR] ANTHROPIC_API_KEY environment variable not set.")
        return None
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        # Claude returns a list of content blocks; join them if needed
        return " ".join([c.text for c in response.content]).strip()
    except Exception as e:
        print(f"[ERROR] Anthropic API call failed: {e}")
        return None

def main():
    # Allow prompt variant selection via command-line argument
    variant = sys.argv[1] if len(sys.argv) > 1 else "base"
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o"
    if variant not in TEMPLATES:
        print(f"Unknown prompt variant '{variant}'. Using 'base'.")
        variant = "base"
    print(f"[Prompt variant: {variant}] [Model: {model}]")
    episode = parse_first_task()
    print("=== ANDROID WORLD AGENT DEMO ===")
    print("Goal:", episode["goal"])
    print("Observation:")
    print(episode["observation"])
    print("\n--- LLM PROMPT ---")
    prompt = format_prompt(episode["goal"], episode["observation"], variant, history=None)
    print(prompt)
    print("--- END PROMPT ---\n")
    # Call LLM
    if model.startswith("claude"):
        action = call_claude(prompt, model=model)
    elif variant == "function_calling":
        action = call_openai_function_calling(prompt, model=model)
    else:
        action = call_openai(prompt, model=model)
    if action:
        print(f"LLM Action: {action}")
    else:
        action = input("Action (manual input): ")
        print(f"You entered: {action}")

if __name__ == "__main__":
    main() 