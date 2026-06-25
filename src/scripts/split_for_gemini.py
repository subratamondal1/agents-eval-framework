import json
import os

input_file = '/Users/subratamondal/Workspace/agents-eval-framework/data/synthetic_voice_sales_traces.json'
output_dir = '/Users/subratamondal/Workspace/agents-eval-framework/data/gemini_prompts'

os.makedirs(output_dir, exist_ok=True)

with open(input_file, 'r') as f:
    data = json.load(f)

prompt_header = """You are an expert conversational AI analyst. 
Below are 30 synthetic call center traces. 
Your task is to classify the difficulty of the conversation based on the transcript.
The difficulty must be exactly one of: ["easy", "standard", "hard", "adversarial"].

Definition of Difficulties:
- easy: Straightforward, highly compliant caller, zero objections, quick booking.
- standard: Normal back-and-forth, minor questions, polite.
- hard: High friction, lots of objections, complex problems, reluctant caller.
- adversarial: Angry, yelling, interrupting, threatening, or completely uncooperative caller.

Output ONLY a raw JSON array of objects, with no markdown formatting and no extra text.
Each object must have exactly two keys: "trace_id" and "difficulty".

Here is the data:
"""

for i in range(10):
    start_idx = i * 30
    end_idx = start_idx + 30
    chunk = data[start_idx:end_idx]
    
    # Trim out unnecessary fields to save Gemini context tokens 
    # (Focus only on what it needs to judge difficulty)
    trimmed_chunk = []
    for trace in chunk:
        trimmed_chunk.append({
            "trace_id": trace["trace_id"],
            "intent": trace["intent"],
            "outcome_booked": trace["outcome_booked"],
            "transcript": trace["transcript"]
        })
        
    out_file = os.path.join(output_dir, f'gemini_prompt_{i+1:02d}.txt')
    with open(out_file, 'w') as f:
        f.write(prompt_header)
        f.write(json.dumps(trimmed_chunk, indent=2))
        
print(f"Successfully created 10 prompt files in {output_dir}")
