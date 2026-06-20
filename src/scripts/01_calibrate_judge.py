import asyncio
import json
import logging
from collections import Counter
from pathlib import Path

import dspy
import pandas as pd
import litellm
import structlog
from dotenv import load_dotenv

logger = structlog.get_logger()

# Live Terminal Tracking for LiteLLM Calls
def track_usage_callback(kwargs, completion_response, start_time, end_time):
    # calculate latency
    if hasattr(end_time, 'timestamp') and hasattr(start_time, 'timestamp'):
        latency_s = end_time.timestamp() - start_time.timestamp()
    else:
        latency_s = float(end_time) - float(start_time)
        
    usage = completion_response.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    
    logger.info(
        "llm_api_call",
        model=kwargs.get("model", ""),
        latency_s=round(latency_s, 2),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=round(kwargs.get("response_cost", 0.0), 6)
    )

litellm.success_callback = [track_usage_callback]

# Add Rate Limit Handling
litellm.num_retries = 10
litellm.request_timeout = 60
litellm.retry_policy = True  # enable exponential backoff

# Hardcoded paths per project rules
TRACES_FILE = Path("/Users/subratamondal/Workspace/agents-eval-framework/data/voice_sales_traces.json")
RESULTS_DIR = Path("/Users/subratamondal/Workspace/agents-eval-framework/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

# Setup LiteLLM through DSPy
lm = dspy.LM("fireworks_ai/accounts/fireworks/models/deepseek-v4-flash")
dspy.settings.configure(lm=lm)


# --- Step 1: Stratification Plan ---

def print_stratification_plan(df: pd.DataFrame) -> None:
    print("\n" + "="*50)
    print("STEP 1: STRATIFICATION PLAN")
    print("="*50)
    
    intent_counts = df['intent'].value_counts()
    
    print("| bucket / intent | trace count | statistical floor |")
    print("|---|---|---|")
    for intent, count in intent_counts.items():
        # Floor target is 48 according to Alex's plan, we just show actual count vs target
        status = "✅ >= 48" if count >= 48 else f"❌ {count}/48"
        print(f"| `{intent}` | {count} | {status} |")
    print("="*50 + "\n")


# --- Step 2: DSPy Judge Calibration ---

class JudgeSignature(dspy.Signature):
    """Evaluate whether an outbound voice sales call successfully resulted in a booked appointment."""
    transcript: str = dspy.InputField(desc="The full transcript of the conversation between the AGENT and the CALLER.")
    rationale: str = dspy.OutputField(desc="One sentence explaining the outcome based on the transcript.")
    is_booked: bool = dspy.OutputField(desc="True if the appointment was successfully booked, False otherwise.")


class VoiceCallJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(JudgeSignature)

    def forward(self, transcript: str):
        return self.predictor(transcript=transcript)


def custom_gepa_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
    """
    Textual feedback metric for GEPA optimization.
    Returns a dspy.Prediction with a scalar 'score' and textual 'feedback'.
    """
    target = bool(gold.outcome_booked)
    predicted = bool(pred.is_booked)
    
    if target == predicted:
        return dspy.Prediction(score=1.0, feedback="Correct.")
    else:
        feedback = f"Mismatch. Ground truth outcome_booked={target}, but model predicted is_booked={predicted}. Rationale given: {pred.rationale}"
        return dspy.Prediction(score=0.0, feedback=feedback)


async def calibrate_judge(train_data: list[dspy.Example]):
    print("STEP 2: CALIBRATING JUDGE (GEPA)")
    
    from dspy.teleprompt import GEPA
    
    # Since we only have a 5-trace sample, we use 'light'.
    optimizer = GEPA(
        metric=custom_gepa_metric,
        auto='light',
        reflection_lm=dspy.settings.lm
    )
    
    student = VoiceCallJudge()
    
    print(f"Starting GEPA optimization over {len(train_data)} examples...")
    compiled_judge = optimizer.compile(
        student=student,
        trainset=train_data
    )
    
    print("Calibration complete. Saving compiled judge...")
    compiled_judge.save(str(RESULTS_DIR / "compiled_judge.json"))
    return compiled_judge


# --- Step 3: Downgrade Headroom Math ---

def calculate_headroom(df: pd.DataFrame, compiled_judge: dspy.Module) -> None:
    print("\n" + "="*50)
    print("STEP 3: DOWNGRADE HEADROOM")
    print("="*50)
    
    #def calculate_headroom(df, compiled_judge):
    from sklearn.metrics import cohen_kappa_score
    results = []
    
    for _, row in df.iterrows():
        prediction = compiled_judge(transcript=row['transcript'])
        
        # Include all original fields from the dataset, plus the new Judge predictions
        result_dict = row.to_dict()
        result_dict["judge_score"] = prediction.is_booked
        result_dict["judge_rationale"] = prediction.rationale
        results.append(result_dict)
        
    results_df = pd.DataFrame(results)
    
    # Calculate Cohen's Kappa
    y_true = results_df['outcome_booked'].astype(bool)
    y_pred = results_df['judge_score'].astype(bool)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    passed_df = results_df[results_df['judge_score'] == True]
    
    print("\n==================================================")
    print("STEP 3: DOWNGRADE HEADROOM & METRICS")
    print("==================================================")
    
    output_path = RESULTS_DIR / "evaluation_traces.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nDetailed judge rationales saved to: {output_path}")
    print(f"Total traces evaluated: {len(results)}")
    print(f"Total traces PASSED by Judge: {len(passed_df)}")
    print(f"Judge Cohen's Kappa Score: {kappa:.4f}")
    
    if kappa < 0.8:
        print(f"WARNING: Kappa score {kappa:.4f} is below target of 0.8!")
    else:
        print(f"SUCCESS: Kappa score {kappa:.4f} meets target of >= 0.8!")
        
    print("\n| trace_id | model_tier_run | intent | cost_total_usd | true_outcome | judge_score |")
    print("|---|---|---|---|---|---|")
    for _, r in results_df.iterrows():
        true_str = "PASS" if r['outcome_booked'] else "FAIL"
        score_str = "PASS" if r['judge_score'] else "FAIL"
        print(f"| `{r['trace_id']}` | {r['model_tier_run']} | `{r['intent']}` | **${r['cost_total_usd']}** | {true_str} | {score_str} |")
    
    print("\nHeadroom math calculation:")
    # We want to find cases where `cheap` handled an intent as successfully as `frontier`
    # For a real dataset we'd average the cost of frontier vs cheap for the same intent.
    
    # Calculate average cost for frontier vs cheap for passed calls
    avg_costs = passed_df.groupby('model_tier_run')['cost_total_usd'].mean()
    
    if 'cheap' in avg_costs and 'frontier' in avg_costs:
        savings_per_call = avg_costs['frontier'] - avg_costs['cheap']
        print(f"\nAverage 'frontier' pass cost: ${avg_costs['frontier']:.4f}")
        print(f"Average 'cheap' pass cost:    ${avg_costs['cheap']:.4f}")
        print(f"Cost savings per call:      ${savings_per_call:.4f}")
        
        total_cheap_passed = len(passed_df[passed_df['model_tier_run'] == 'cheap'])
        total_headroom = savings_per_call * total_cheap_passed
        print(f"\nTotal Dollar Headroom Extrapolated for dataset: **${total_headroom:.4f}**")
    else:
        print("\nNot enough data points in sample to compare frontier vs cheap directly.")


async def main():
    import random
    
    # 1. Load Data
    with open(TRACES_FILE, 'r') as f:
        traces = json.load(f)
        
    df = pd.DataFrame(traces)
    
    # Step 1
    print_stratification_plan(df)
    
    # 2. Build DSPy Example instances
    all_data = []
    for t in traces:
        all_data.append(dspy.Example(
            transcript=t["transcript"],
            outcome_booked=t["outcome_booked"]
        ).with_inputs("transcript"))
        
    # Sample a maximum of 20 traces for GEPA optimization to save API credits.
    # The final headroom calculation will still run over all 480 traces.
    random.seed(42)
    train_data = random.sample(all_data, min(20, len(all_data)))
        
    # Step 2
    compiled_judge = await calibrate_judge(train_data)
    
    # Step 3
    calculate_headroom(df, compiled_judge)
    
    # Save Raw DSPy LLM History as Clean JSON
    history_path = RESULTS_DIR / "dspy_raw_llm_history.json"
    clean_history = []
    for entry in lm.history:
        clean_history.append({
            "messages": entry.get("messages", []),
            "response_text": entry.get("outputs", [{}])[0].get("text", "") if "outputs" in entry else str(entry.get("response", ""))
        })
        
    with open(history_path, "w") as f:
        json.dump(clean_history, f, indent=2)
    print(f"Clean LLM Prompt/Response history saved to: {history_path}")

if __name__ == "__main__":
    asyncio.run(main())
