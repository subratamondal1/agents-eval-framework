# DSPy Voice Agent Eval Benchmark

**A production-grade, κ-calibrated LLM-as-Judge evaluation framework for voice AI agents** — with autonomous prompt optimization, cost-aware model routing, and a rigorous benchmark versioning methodology that iterates from κ=0.12 to κ=0.93.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue?style=flat)](https://python.org)
[![DSPy](https://img.shields.io/badge/framework-DSPy-purple?style=flat)](https://dspy.ai)
![Judge Calibration](https://img.shields.io/badge/Cohen's%20κ-0.9261-brightgreen?style=flat)
![Cost per 480 traces](https://img.shields.io/badge/eval%20cost-$0.15%20%2F%20480%20traces-green?style=flat)

---

## What this is

This repository is a benchmark design system, not just evaluation code. The goal was to prove that an LLM judge can reliably evaluate voice agent call outcomes (did the call result in a booking?) with calibrated, human-verified accuracy — at a cost that makes continuous evaluation economically viable in production.

The hard part was not writing a prompt. The hard part was designing the benchmark so the judge's κ score represented genuine semantic reasoning rather than pattern-matching artifacts from label leakage. That took five architectural iterations and is documented in full below.

---

## Benchmark results

### Final eval (V5): 480-trace dataset

| Metric | Result |
|---|---|
| Total traces evaluated | 480 |
| Judge calibration (Cohen's κ) | **0.9261** |
| Total eval cost | **$0.149** (15 cents) |
| Projected cost per 10,000 calls | **$3.10** |
| FinOps headroom unlocked per 10K calls | **$217.00** |

### Architectural versioning — how κ went from 0.12 to 0.93

This table is the core research contribution of this repo. Each version represents a distinct architectural hypothesis, a failure mode discovered, and the insight that drove the next iteration.

| Version | Architecture | Model | Cohen's κ | Verdict |
|---|---|---|---|---|
| V1 | Single-pass, **unblinded** (label leakage) | deepseek-v4-flash | 1.0000 | ❌ Artificial — judge read the `[OUTCOME]` tag |
| V2 | Single-pass DSPy signature, blinded | deepseek-v4-flash | 0.1245 | ❌ Failed — no structural separation of extraction and judgment |
| V3 | Single-pass DSPy signature, blinded | deepseek-v4-pro | 0.0130 | ❌ Failed — context collapse on 2,000+ token transcripts |
| V4 | Multi-agent: Extractor → Judge | deepseek-v4-flash | 0.2948 | ❌ Failed — extractor fidelity too low; garbage in, garbage out |
| **V5** | Multi-agent + dual-stratified 40-trace matrix | deepseek-v4-flash | **0.9261** | ✅ **SUCCESS — exceeds 0.80 target** |

---

## Failure analysis: what each version revealed

### V1 — Label leakage gives you a perfect but useless judge (κ = 1.0)

The first pass achieved a perfect κ score. It was immediately suspicious. Diagnosis: the raw `[OUTCOME]` tag was present in the transcript field fed to the judge. The judge was not evaluating call quality — it was reading the label. κ = 1.0 was noise masquerading as signal.

**Fix:** All outcome-related fields are programmatically stripped before the transcript enters the DSPy context window. The fields excluded from judge context are: `trace_id`, `model_tier_run`, `outcome_booked`, `roi_usd`, `cost_total_usd`, and all cost telemetry columns.

### V2 — Single-pass DSPy signature cannot hold two cognitive tasks (κ = 0.12)

After blinding, the same single-pass architecture collapsed. A single DSPy signature was asked to simultaneously extract structured conversational state from a 2,000+ token transcript and evaluate boolean booking logic. The model conflated the two tasks.

**Fix:** Decompose into two nodes. Node 1 (Extractor) extracts clean conversational state. Node 2 (Judge) evaluates boolean logic over the clean state. This is the Extractor → Judge graph that all subsequent versions use.

### V3 — A stronger model made it worse (κ = 0.01)

Switching from `deepseek-v4-flash` to `deepseek-v4-pro` in the single-pass architecture made κ collapse further. The more capable model attempted deeper reasoning over the raw transcript, which caused context collapse on long inputs — it attended to the wrong parts of the transcript and produced incoherent rationales.

**Insight:** Model capability does not substitute for structural decomposition. A more capable model given a malformed task fails more dramatically, not less.

### V4 — Multi-agent architecture is necessary but not sufficient (κ = 0.29)

The Extract → Judge split improved κ from 0.01 to 0.29. But extractor fidelity was still too low: the extractor was producing inconsistent state representations across different intent types (objection handling produced different output shapes than appointment booking). Garbage in, garbage out.

**Fix:** Stratify the training data by intent class before optimization. The GEPA loop needs to see representative failure cases from each intent category, not a random sample dominated by the most common class.

### V5 — Dual-stratified training matrix achieves target κ (κ = 0.93)

Training the GEPA optimizer on a 40-trace matrix — 4 traces × 10 intent classes — gave it balanced coverage of every conversational pattern. The judge's compiled prompt now handles all 10 intent types reliably. κ = 0.9261 against human annotations, exceeding the 0.80 target.

---

## Intent taxonomy

The benchmark stratifies voice call transcripts into **10 intent classes**, 48 traces each, for a total of 480 evaluation traces:

| # | Intent class | Description |
|---|---|---|
| 1 | `appointment_booking` | Caller explicitly books or confirms a time slot |
| 2 | `pricing_question` | Caller asks about cost before committing |
| 3 | `objection_handling` | Caller raises resistance; agent must address it |
| 4 | `ambiguous_intent` | Caller's goal is unclear; tests judge's uncertainty handling |
| 5 | `product_inquiry` | Caller asks about service scope, not price |
| 6 | `lead_qualification` | Agent probes for decision-making authority and timeline |
| 7 | `complaint_de_escalation` | Caller starts hostile; agent attempts recovery |
| 8 | `interruption_overlap` | Talk-over patterns; tests transcript parsing fidelity |
| 9 | `upsell_attempt` | Agent attempts to expand scope mid-call |
| 10 | `callback_scheduling` | No booking today; caller schedules a follow-up |

**Why taxonomy matters for evaluation:** A judge trained on a random transcript sample will be dominated by the most common intent class and fail on edge cases. The stratified 40-trace training matrix (4 per class) forces the GEPA optimizer to learn patterns for all 10 types. This is why V4 (unstratified) gave κ = 0.29 and V5 (stratified) gave κ = 0.93.

---

## Architecture

### The Extractor → Judge graph (DSPy)

```
transcript (blinded)
       │
       ▼
  ┌──────────┐
  │ Extractor│  ← deepseek-v4-pro (heavy extraction, cognitive load high)
  │  Node 1  │  extracts: booking_signal, objection_count, commitment_language
  └──────────┘
       │ clean structured state
       ▼
  ┌──────────┐
  │  Judge   │  ← deepseek-v4-flash (boolean logic, cognitive load low)
  │  Node 2  │  outputs: reasoning (CoT), rationale (1 sentence), is_booked (bool)
  └──────────┘
```

### FinOps routing

Models are routed by cognitive load, not by default. Heavy extraction (understanding 2,000+ token transcripts) goes to `deepseek-v4-pro`. Simple boolean evaluation over a structured state goes to `deepseek-v4-flash`. The cost breakdown for 480 traces:

| Cost component | Tokens | Rate | Cost |
|---|---|---|---|
| Input (transcripts, extraction) | ~851,000 | $0.14 / 1M | $0.119 |
| Output (rationale, boolean) | ~38,400 | $0.28 / 1M | $0.010 |
| GEPA optimization overhead | 20 samples | — | $0.020 |
| **Total** | | | **$0.149** |

At this rate, evaluating 10,000 production calls costs **$3.10** — compared to **$217.00** using a monolithic frontier model prompt. The architecture pays for itself at any scale above a few hundred calls per day.

### The GEPA optimization loop

GEPA (Generate, Evaluate, Propose, Accept) is an offline prompt optimization loop that replaces manual prompt engineering:

1. **Generate** — run the judge on a training sample; collect predictions.
2. **Evaluate** — score each prediction against the ground truth CRM outcome.
3. **Propose** — parse failure traces; identify the patterns the judge is missing; rewrite the system prompt to address them.
4. **Accept** — if the proposed prompt improves κ on the held-out set, accept it.

The output is a deterministic DSPy-compiled prompt with strict output typing — no free-form generation at inference time.

### DSPy compiled prompt (V5, GEPA-derived)

```text
Your input fields are:
1. `transcript` (str): The full transcript of the conversation between the AGENT and the CALLER.

Your output fields are:
1. `reasoning` (str): [Chain of Thought — injected automatically by DSPy]
2. `rationale` (str): One sentence explaining the outcome based on the transcript.
3. `is_booked` (bool): True if the appointment was successfully booked, False otherwise.

[[ ## transcript ## ]]
{transcript}

[[ ## reasoning ## ]]
{reasoning}

[[ ## rationale ## ]]
{rationale}

[[ ## is_booked ## ]]
{is_booked}   # must be True or False

[[ ## completed ## ]]
Objective: Evaluate whether an outbound voice sales call successfully resulted in a booked appointment.
```

---

## Blind evaluation protocol

To ensure κ measures semantic reasoning and not artifact-matching, every transcript is blinded before entering the DSPy context window.

**Included in judge context:**
- `transcript` — with `[OUTCOME]` tag programmatically stripped

**Excluded from judge context (all of these):**
- `trace_id`, `model_tier_run`, `outcome_booked`
- `roi_usd`, `cost_total_usd`
- `cost_telephony_usd`, `cost_transcription_usd`, `cost_llm_usd`

The κ score is computed between judge predictions and the true CRM `outcome_booked` field — which the judge never sees.

---

## Quickstart

```bash
git clone https://github.com/subratamondal1/agents-eval-framework
cd agents-eval-framework
uv sync   # or: pip install -r requirements.txt

# Run the eval benchmark
python eval/run_benchmark.py

# Run the GEPA optimization loop
python eval/gepa_optimize.py
```

Set your API keys in `.env`:
```bash
DEEPSEEK_API_KEY=your_key_here
FIREWORKS_API_KEY=your_key_here   # optional: for Fireworks-hosted DeepSeek
```

---

## Key lessons

1. **Label leakage gives you a perfect but useless judge.** Always strip outcome fields before they touch the judge context. κ = 1.0 from an unblinded judge is a red flag, not a success.
2. **Model capability does not substitute for structural decomposition.** A stronger model on a badly-framed task fails harder. V3 (deepseek-v4-pro, single-pass) gave κ = 0.01.
3. **Stratified training matters as much as architecture.** V4 and V5 use the same Extract → Judge graph. The difference is stratified training data. κ went from 0.29 to 0.93.
4. **Cohen's κ is the right metric for judge calibration** — not accuracy, not F1. κ corrects for chance agreement, which matters when class imbalance is present (most calls do not result in a booking).
5. **FinOps routing by cognitive load, not by default.** Sending everything to the most capable model is financially disastrous at scale and produces worse results on long-context tasks due to context collapse.

---

## License

MIT — see [LICENSE](LICENSE).
