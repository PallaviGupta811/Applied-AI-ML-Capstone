# Part 4 — LLM-Powered Model Explanation Pipeline

## Overview

This part adds an LLM-powered explanation layer on top of the machine learning model developed in Part 3.

The objective is to combine predictive machine learning with natural language explanations by using an LLM to explain why a model made a particular prediction.

---

# Selected Track

## Track C — Model Prediction Explanation Pipeline

For this implementation:

- The best-performing model from Part 3 (`best_model.pkl`) is loaded.
- Three feature-vector inputs are passed through the model.
- The model generates:
  - Predicted class
  - Prediction probability
- These outputs are sent to an LLM.
- The LLM returns a structured JSON explanation.
- The output is validated using JSON schema validation.

---

# System Architecture

Input Features
|
↓
Saved Random Forest Model
(best_model.pkl)
|
↓
Prediction + Probability
|
↓
Groq LLM API
|
↓
Structured JSON Explanation
|
↓
Schema Validation


---

# Technologies Used

- Python
- Scikit-learn
- Joblib
- Requests
- Groq API
- Llama 3.1 Model
- JSON Schema Validation
- Regular Expressions

---

# LLM API Setup

The Groq API was used through HTTP POST requests.

The API key was stored securely using an environment variable:


GROQ_API_KEY


The API key was loaded using:

```python
load_dotenv()

No API keys were hardcoded in the notebook or repository.

LLM Model Used

Model:

llama-3.1-8b-instant

The model was accessed using the Groq API.

Prompt Design
System Prompt
You are a machine learning explanation assistant.

Your task is to explain house price classification predictions.

Return ONLY valid JSON.
Do not use markdown.
Do not add extra text.

The JSON must contain:
prediction_label,
confidence_level,
top_reason,
second_reason,
next_step
User Prompt Template
Important house features:
{features}

Model prediction:
{prediction}

Prediction probability:
{probability}

Rules:

prediction_label:
Return only "High Price" or "Low Price"

confidence_level:
Return only "low", "medium", or "high"

Explain the prediction briefly.

Return JSON only.
Model Prediction Pipeline

The saved model from Part 3 was loaded:

joblib.load("best_model.pkl")

For each input:

Feature vector was prepared.
Model prediction was generated.
Prediction probability was calculated using:
predict_proba()
The prediction details were sent to the LLM.
Structured Output Validation

The LLM output was required to follow this JSON schema:

Field	Type
prediction_label	String
confidence_level	String
top_reason	String
second_reason	String
next_step	String

Validation was performed using:

jsonschema.validate()

If validation failed, a fallback response was returned.

Prediction Explanation Results
Input	Predicted Class	Probability	Validation Status
Record 1	1	0.88	Pass
Record 2	0	0.04	Pass
Record 3	1	0.735	Pass

Example output:

{
  "prediction_label": "High Price",
  "confidence_level": "high",
  "top_reason": "High Lot Frontage and Overall Quality indicate a high-end property",
  "second_reason": "Year Built and renovation features influence the prediction",
  "next_step": "Verify property details before final decision"
}
Temperature Experiment

The LLM was tested with two temperature values:

Temperature = 0
Temperature = 0.7
Comparison
Temperature	Behaviour
0	More deterministic and consistent responses
0.7	More variation and creative responses

Temperature 0 was selected for the main pipeline because structured generation tasks require predictable outputs.

A higher temperature introduces randomness by allowing the model to select from a wider range of possible responses.

Safety Guardrails

A PII detection layer was implemented before every LLM request.

The system checks for:

Email addresses
Phone numbers

using regular expressions.

Test Results
Blocked Input
example@gmail.com

Output:

Input blocked: PII detected.
Allowed Input
Explain this house prediction

Output:

LLM response generated successfully.