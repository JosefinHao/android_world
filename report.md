# QualGent Research Coding Challenge: Short Report (10 Episodes)

## Approach to Prompting and Evaluation

- **Prompting:**  
  Used three prompt variants: base, few-shot, and self-reflection. Prompts provided the goal and current observation, and explicitly instructed the LLM to select from available UI elements.
- **Agent Loop:**  
  For each episode, the agent receives the goal and observation, and generates the next action using an LLM (GPT-4o, GPT-3.5-turbo, Claude-3-opus-20230229).
- **Evaluation:**  
  Ran the agent on **10 episodes per model**, 3 steps per episode. Compared LLM actions to ground-truth. Logged step accuracy, episode success, and hallucinated actions.

---

## Summary of Performance Metrics

| Model                  | Step Accuracy | Episode Success | Hallucinated Actions |
|------------------------|---------------|----------------|----------------------|
| GPT-4o                 | 10/30 = 0.33  | 0/10 = 0.00    | 2/30 = 0.07          |
| GPT-3.5-turbo          | 8/30 = 0.27   | 0/10 = 0.00    | 0/30 = 0.00          |
| Claude-3-opus-20240229 | 0/30 = 0.00   | 0/10 = 0.00    | 26/30 = 0.87         |

- **Episodes evaluated:** 10 per model
- **Steps per episode:** 3

---

## Illustrative Example Episodes

### Example Episode 1
**Goal:** What events do I have October 26 2023 in Simple Calendar Pro?

| Step | Model                  | LLM Action         | Ground Truth        | Correct? | Hallucination? |
|------|------------------------|--------------------|--------------------|----------|----------------|
| 1    | GPT-4o                 | CLICK("Apps")      | CLICK("Apps")      | ✅       |                |
| 1    | GPT-3.5-turbo          | CLICK("Apps")      | CLICK("Apps")      | ✅       |                |
| 1    | Claude-3-opus-20240229 | None               | CLICK("Apps")      | ❌       | ✅             |
| 2    | GPT-4o                 | CLICK("Step2")     | CLICK("Apps")      | ❌       |                |
| 2    | GPT-3.5-turbo          | CLICK("Data Dive") | CLICK("Apps")      | ❌       |                |
| 2    | Claude-3-opus-20240229 | None               | CLICK("Apps")      | ❌       | ✅             |
| 3    | GPT-4o                 | CLICK("Step3")     | CLICK("Data Dive") | ❌       |                |
| 3    | GPT-3.5-turbo          | CLICK("Settings")  | CLICK("Data Dive") | ❌       |                |
| 3    | Claude-3-opus-20240229 | None               | CLICK("Data Dive") | ❌       | ✅             |

**Episode success:** All models: False

---

### Example Episode 2
**Goal:** What is my next upcoming event in Simple Calendar Pro?

| Step | Model                  | LLM Action         | Ground Truth        | Correct? | Hallucination? |
|------|------------------------|--------------------|--------------------|----------|----------------|
| 1    | GPT-4o                 | CLICK("Apps")      | CLICK("Apps")      | ✅       |                |
| 1    | GPT-3.5-turbo          | CLICK("Apps")      | CLICK("Apps")      | ✅       |                |
| 1    | Claude-3-opus-20240229 | None               | CLICK("Apps")      | ❌       | ✅             |
| 2    | GPT-4o                 | CLICK("Step2")     | CLICK("Apps")      | ❌       |                |
| 2    | GPT-3.5-turbo          | CLICK("Data Dive") | CLICK("Apps")      | ❌       |                |
| 2    | Claude-3-opus-20240229 | None               | CLICK("Apps")      | ❌       | ✅             |
| 3    | GPT-4o                 | CLICK("Step3")     | CLICK("Data Dive") | ❌       |                |
| 3    | GPT-3.5-turbo          | CLICK("Settings")  | CLICK("Data Dive") | ❌       |                |
| 3    | Claude-3-opus-20240229 | None               | CLICK("Data Dive") | ❌       | ✅             |

**Episode success:** All models: False

---

### Example Episode 3
**Goal:** What is the location of my Movie night event in Simple Calendar Pro? Answer with the location only.

| Step | Model                  | LLM Action         | Ground Truth        | Correct? | Hallucination? |
|------|------------------------|--------------------|--------------------|----------|----------------|
| 1    | GPT-4o                 | Sarah's apartment  | CLICK("Apps")      | ❌       | ✅             |
| 1    | GPT-3.5-turbo          | CLICK("Step1")     | CLICK("Apps")      | ❌       |                |
| 1    | Claude-3-opus-20240229 | None               | CLICK("Apps")      | ❌       | ✅             |
| 2    | GPT-4o                 | CLICK("Step2")     | CLICK("Apps")      | ❌       |                |
| 2    | GPT-3.5-turbo          | CLICK("Step2")     | CLICK("Apps")      | ❌       |                |
| 2    | Claude-3-opus-20240229 | None               | CLICK("Apps")      | ❌       | ✅             |
| 3    | GPT-4o                 | CLICK("Step3")     | CLICK("Data Dive") | ❌       |                |
| 3    | GPT-3.5-turbo          | CLICK("Step3")     | CLICK("Data Dive") | ❌       |                |
| 3    | Claude-3-opus-20240229 | None               | CLICK("Data Dive") | ❌       | ✅             |

**Episode success:** All models: False

---

## Recommendations for Improving Agent Behavior

- Incorporate more advanced search or retrieval mechanisms to help the agent find relevant information in complex or lengthy observations.
- Implement retry logic or self-correction strategies when the agent produces invalid or hallucinated actions.
- Refine prompt engineering, including clearer instructions and more targeted few-shot examples, to reduce ambiguity and improve action selection.
- Explore dynamic adjustment of prompt context (e.g., selectively including only the most relevant history or observations).
- Investigate hybrid approaches, such as combining LLMs with rule-based or symbolic reasoning for better reliability.
- Evaluate the impact of different temperature and decoding settings on agent consistency and accuracy.

---

## Bonus Features

- **Memory Buffer:** The agent loop and prompt formatting support including the full history of actions and observations in the prompt, enabling the LLM to reason over multi-step episodes.
- **Multi-Model Comparison:** The framework benchmarks and compares multiple models (GPT-4o, GPT-3.5-turbo, Claude-3-opus-20240229). While Mistral is not included, the code is easily extensible to add more models.
- **Visualization:** A Streamlit dashboard is provided for interactive visualization of episode progress, step-by-step results, and model comparison.

---

## Using the Streamlit Dashboard for Visualization

To interactively explore and visualize the evaluation results, launch the Streamlit dashboard with the following command:

```
streamlit run src/streamlit_app.py
```

This will open a local web page (usually at http://localhost:8501) where you can:
- Compare model performance (step accuracy, hallucination rate, episode success rate)
- Drill down into individual episodes and steps for each model
- Filter for incorrect or hallucinated steps
- Download results as CSV for further analysis

The dashboard provides an intuitive way to analyze and present the benchmarking results. 