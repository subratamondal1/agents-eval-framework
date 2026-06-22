import asyncio
import json
from pathlib import Path
import re

import dspy
import structlog
from dotenv import load_dotenv

logger = structlog.get_logger()

# Setup LiteLLM through DSPy
load_dotenv()
lm = dspy.LM("fireworks_ai/accounts/fireworks/models/deepseek-v4-flash")
dspy.settings.configure(lm=lm)

class ExtractIntentSignature(dspy.Signature):
    """Extract the primary intent and caller engagement from the transcript."""
    transcript: str = dspy.InputField(desc="The full transcript of the conversation between the AGENT and the CALLER.")
    agent_intent: str = dspy.OutputField(desc="The core goal the agent is attempting in this call.")
    caller_reaction: str = dspy.OutputField(desc="How the caller responded to the agent.")

class JudgeBookingSignature(dspy.Signature):
    """Evaluate whether an outbound voice sales call resulted in a booked appointment based on extracted context."""
    agent_intent: str = dspy.InputField(desc="The core goal the agent is attempting in this call.")
    caller_reaction: str = dspy.InputField(desc="How the caller responded to the agent.")
    rationale: str = dspy.OutputField(desc="One sentence explaining the outcome based on the intent and reaction.")
    is_booked: bool = dspy.OutputField(desc="True if the appointment was successfully booked, False otherwise.")

class VoiceCallJudgeExport(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(ExtractIntentSignature)
        self.judge = dspy.ChainOfThought(JudgeBookingSignature)

    def forward(self, transcript: str):
        context = self.extractor(transcript=transcript)
        judgment = self.judge(agent_intent=context.agent_intent, caller_reaction=context.caller_reaction)
        
        return dspy.Prediction(
            agent_intent=context.agent_intent,
            caller_reaction=context.caller_reaction,
            rationale=judgment.rationale,
            is_booked=judgment.is_booked
        )

async def main():
    TRACES_FILE = Path("/Users/subratamondal/Workspace/agents-eval-framework/data/voice_sales_traces.json")
    COMPILED_JUDGE_PATH = Path("/Users/subratamondal/Workspace/agents-eval-framework/results/run_20260622_123218_blinded/compiled_judge.json")
    OUTPUT_PATH = Path("/Users/subratamondal/Workspace/agents-eval-framework/results/run_20260622_123218_blinded/mapped_decisions_v4.json")
    
    with open(TRACES_FILE, 'r') as f:
        traces = json.load(f)
        
    def strip_outcome_leakage(transcript: str) -> str:
        return re.sub(r'\s*\[OUTCOME\]\s*(booked|no_booking)', '', transcript).strip()
        
    compiled_judge = VoiceCallJudgeExport()
    compiled_judge.load(str(COMPILED_JUDGE_PATH))
    
    print(f"Running inference on {len(traces)} traces to map intermediate state. This may take a few minutes...")
    
    async def process_trace(row):
        clean_transcript = strip_outcome_leakage(row["transcript"])
        # Because we already ran this prompt over these transcripts in the previous script,
        # DSPy will hit its local SQLite cache instantly for 100% of these calls!
        # It will take seconds, not minutes.
        pred = await asyncio.to_thread(compiled_judge, transcript=clean_transcript)
        
        mapped = row.copy()
        mapped["extractor_intent"] = pred.agent_intent
        mapped["extractor_reaction"] = pred.caller_reaction
        mapped["judge_rationale"] = pred.rationale
        mapped["judge_prediction"] = pred.is_booked
        return mapped
        
    # High concurrency because 100% should be cache hits.
    semaphore = asyncio.Semaphore(50)
    async def sem_process(row):
        async with semaphore:
            return await process_trace(row)
            
    tasks = [sem_process(row) for row in traces]
    results = await asyncio.gather(*tasks)
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Successfully mapped decisions to {len(results)} traces. Saved at {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
