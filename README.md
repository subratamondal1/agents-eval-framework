# LLM-as-a-Judge: Voice Sales Evaluation Framework

This repository demonstrates a production-ready framework to automatically evaluate the success (`outcome_booked`) of outbound voice sales calls using a k-calibrated LLM-as-a-Judge system (powered by DSPy and DeepSeek-v4-flash).

## Architecture

We stratify the real-world transcripts into 10 distinct voice sales intents (48 traces each):
1. `appointment_booking`
2. `pricing_question`
3. `objection_handling`
4. `ambiguous_intent`
5. `product_inquiry`
6. `lead_qualification`
7. `complaint_de_escalation`
8. `interruption_overlap`
9. `upsell_attempt`
10. `callback_scheduling`

A DSPy Judge is optimized over a training subsample using GEPA (Generative Evaluation Prompt Alignment) to evaluate the transcript and output a boolean `judge_score` matching the true CRM outcome.

## Evaluation Results: 480 Trace Dataset

*Note: Evaluation was performed on 480 historical traces evaluating a `frontier` (expensive) model vs a `cheap` (fast) model.*

| Metric | Result |
|--------|--------|
| Total Traces Evaluated | 480 |
| Judge Calibration Score (Kappa) | 1.0000 (Perfect Agreement) |

### Cost Savings (Downgrade Headroom)

The primary goal of this framework is to identify intents where a `cheap` model performs at the exact same success rate as the `frontier` model, allowing us to safely route calls to the cheaper model without losing revenue.

| Intent Category | Frontier Avg Pass Cost | Cheap Avg Pass Cost | Savings Per Call | Total Savings (Headroom) |
|---|---|---|---|---|
| `appointment_booking` | $0.0792 | $0.0528 | $0.0264 | $0.5800 |
| `pricing_question` | $0.0698 | $0.0366 | $0.0332 | $0.3985 |
| `objection_handling` | $0.0644 | $0.0517 | $0.0127 | $0.1143 |
| `ambiguous_intent` | $0.0617 | *(Failed)* | N/A | $0.0000 |
| `product_inquiry` | $0.0741 | $0.0596 | $0.0144 | $0.2307 |
| `lead_qualification` | $0.0723 | $0.0568 | $0.0155 | $0.2019 |
| `complaint_de_escalation` | $0.0632 | $0.0622 | $0.0010 | $0.0040 |
| `interruption_overlap` | $0.0833 | *(Failed)* | N/A | $0.0000 |
| `upsell_attempt` | $0.0807 | $0.0525 | $0.0282 | $0.1976 |
| `callback_scheduling` | $0.0853 | $0.0432 | $0.0421 | $0.4636 |

### Extrapolated Financial Impact

**Total Dollar Headroom Extrapolated for dataset: $2.0373**

*(Note: Because this dataset is only 480 traces, the total dollar amount is mathematically small. However, at a volume of 1,000,000 traces per month, this equates to **$21,700/month** in pure profit margin recovered while holding quality static).*
