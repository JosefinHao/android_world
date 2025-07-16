import streamlit as st
import json
from pathlib import Path
import pandas as pd
import altair as alt

RESULTS_FILE = Path(__file__).parent.parent / "results" / "eval_log.jsonl"

@st.cache_data
def load_data():
    episodes_by_model = {}
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            model = entry.get("model", "unknown")
            if model not in episodes_by_model:
                episodes_by_model[model] = {}
            ep = entry["episode"]
            if ep not in episodes_by_model[model]:
                episodes_by_model[model][ep] = []
            episodes_by_model[model][ep].append(entry)
    return episodes_by_model

def get_model_metrics(episodes):
    all_steps = [s for ep in episodes.values() for s in ep]
    total_steps = len(all_steps)
    correct_steps = sum(1 for s in all_steps if s["correct"])
    hallucinations = sum(1 for s in all_steps if s["hallucination"])
    episode_success = sum(1 for ep in episodes.values() if all(s["correct"] for s in ep))
    n_episodes = len(episodes)
    return {
        "total_steps": total_steps,
        "correct_steps": correct_steps,
        "hallucinations": hallucinations,
        "episode_success": episode_success,
        "n_episodes": n_episodes,
        "step_accuracy": correct_steps / total_steps if total_steps else 0,
        "hallucination_rate": hallucinations / total_steps if total_steps else 0,
        "episode_success_rate": episode_success / n_episodes if n_episodes else 0,
    }

def main():
    st.title("Android World LLM Agent Evaluation Results")
    episodes_by_model = load_data()
    models = list(episodes_by_model.keys())
    # --- Plots: Step accuracy and hallucination rate per model ---
    plot_data = []
    for model in models:
        metrics = get_model_metrics(episodes_by_model[model])
        plot_data.append({
            "Model": model,
            "Step Accuracy": metrics["step_accuracy"],
            "Hallucination Rate": metrics["hallucination_rate"],
            "Episode Success Rate": metrics["episode_success_rate"]
        })
    plot_df = pd.DataFrame(plot_data)
    st.write("### Model Comparison")
    st.altair_chart(
        alt.Chart(plot_df).mark_bar().encode(
            x=alt.X('Model', sort=None),
            y=alt.Y('Step Accuracy', scale=alt.Scale(domain=[0,1])),
            color=alt.value('steelblue'),
            tooltip=['Model', 'Step Accuracy']
        ).properties(title="Step Accuracy per Model"),
        use_container_width=True
    )
    st.altair_chart(
        alt.Chart(plot_df).mark_bar().encode(
            x=alt.X('Model', sort=None),
            y=alt.Y('Hallucination Rate', scale=alt.Scale(domain=[0,1])),
            color=alt.value('orange'),
            tooltip=['Model', 'Hallucination Rate']
        ).properties(title="Hallucination Rate per Model"),
        use_container_width=True
    )
    st.altair_chart(
        alt.Chart(plot_df).mark_bar().encode(
            x=alt.X('Model', sort=None),
            y=alt.Y('Episode Success Rate', scale=alt.Scale(domain=[0,1])),
            color=alt.value('green'),
            tooltip=['Model', 'Episode Success Rate']
        ).properties(title="Episode Success Rate per Model"),
        use_container_width=True
    )
    # --- Per-episode/step view ---
    model = st.selectbox("Select model", models)
    episodes = episodes_by_model[model]
    episode_nums = sorted(episodes.keys())
    episode = st.selectbox("Select episode", episode_nums)
    steps = episodes[episode]
    st.write(f"### Model: {model} | Episode: {episode}")
    st.write(f"Goal: {steps[0]['goal']}")
    # Filters
    show_only_incorrect = st.checkbox("Show only incorrect steps", value=False)
    show_only_hallucinations = st.checkbox("Show only hallucinations", value=False)
    # Table
    st.write("#### Steps:")
    for step in steps:
        if show_only_incorrect and step["correct"]:
            continue
        if show_only_hallucinations and not step["hallucination"]:
            continue
        color = "#d4edda" if step["correct"] else ("#f8d7da" if not step["correct"] else "white")
        if step["hallucination"]:
            color = "#fff3cd"  # yellow for hallucination
        st.markdown(f"<div style='background-color:{color};padding:8px;border-radius:6px'>"
                    f"<b>Step {step['step']}</b><br>"
                    f"<b>Observation:</b> {step['observation']}<br>"
                    f"<b>LLM Action:</b> {step['llm_action']}<br>"
                    f"<b>Ground Truth:</b> {step['ground_truth']}<br>"
                    f"<b>Correct:</b> {step['correct']} | <b>Hallucination:</b> {step['hallucination']}<br>"
                    f"<b>Clicked Element:</b> {step['clicked_elem']} | <b>UI Elements:</b> {step['ui_elements']}"
                    f"</div>", unsafe_allow_html=True)
    # Summary metrics
    total_steps = len(steps)
    correct_steps = sum(1 for s in steps if s["correct"])
    hallucinations = sum(1 for s in steps if s["hallucination"])
    episode_success = all(s["correct"] for s in steps)
    st.write(f"#### Episode Summary:")
    st.write(f"Step accuracy: {correct_steps}/{total_steps} = {correct_steps/total_steps:.2f}")
    st.write(f"Episode success: {episode_success}")
    st.write(f"Hallucinated actions: {hallucinations}/{total_steps} = {hallucinations/total_steps:.2f}")
    # Overall summary for model
    all_steps = [s for ep in episodes.values() for s in ep]
    total_steps_all = len(all_steps)
    correct_steps_all = sum(1 for s in all_steps if s["correct"])
    hallucinations_all = sum(1 for s in all_steps if s["hallucination"])
    episode_success_all = sum(1 for ep in episodes.values() if all(s["correct"] for s in ep))
    n_episodes = len(episodes)
    st.write(f"#### Model Summary:")
    st.write(f"Total episodes: {n_episodes}")
    st.write(f"Total steps: {total_steps_all}")
    st.write(f"Step accuracy: {correct_steps_all}/{total_steps_all} = {correct_steps_all/total_steps_all:.2f}")
    st.write(f"Episode success rate: {episode_success_all}/{n_episodes} = {episode_success_all/n_episodes:.2f}")
    st.write(f"Hallucinated actions: {hallucinations_all}/{total_steps_all} = {hallucinations_all/total_steps_all:.2f}")
    # --- CSV Export ---
    df = pd.DataFrame(all_steps)
    st.download_button(
        label="Download all steps for this model as CSV",
        data=df.to_csv(index=False),
        file_name=f"{model}_results.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main() 