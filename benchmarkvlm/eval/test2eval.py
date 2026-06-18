from openrouter import OpenRouter
import pandas as pd
import json
import base64
import os
import time


# CLIENT OPENROUTER

or_client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

# Objects and scene 

objects = [
    "blue pen",
    "grey mug",
    "computer mouse",
    "orange balloon",
    "blue can",
    "USB drive",
    "whiteboard eraser",
    "stirrer"
]

scene_image = "benchmarkvlm/data/datatest1,2,3/everything.jpeg"

# MAPPING MODELS 

model_mapping = {
    "GPT-5.4 Mini": "openai/gpt-5.4-mini",
    "Qwen 3.5 9B": "qwen/qwen3.5-9b",
    "Gemini 3.1 Flash Image": "google/gemini-3.1-flash-image-preview",
    "Ministral 8B": "mistralai/ministral-8b-2512",
    "Seed 2.0 Lite": "bytedance-seed/seed-2.0-lite"
}


# Encode image in base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

# Safe JSON parse

def safe_json_parse(text):
    if not text:
        return {"error": "empty_response", "raw": text}

    try:
        return json.loads(text)
    except:
        pass
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
    except:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
    except:
        pass

    return {"error": "invalid_json", "raw": text}

# Function

def evaluate_response(object_name, vlm_response, model_id="anthropic/claude-sonnet-4.6"):

    images_base64 = [encode_image(path) for path in [scene_image]]

    # Prompt 
    prompt = f"""
You are a robotics perception and manipulation expert.

You are given an image with multiple objects (attached below) and a VLM's response about this object.

Your task is to **evaluate the VLM response** according to the following **Robot-Centric Scoring (V5)** protocol:

 Evaluation Protocol — Robot-Centric Scoring (V5)
Conditional Evaluation Rule:
- If identification is incorrect, Planning, Grasp, and Use are evaluated with respect to the object as described by the model.
- Physically inconsistent actions are penalized.
- Identification errors do not propagate to other criteria.

Scoring Overview (0–5 scale):

Identification (A /5)
- Position: +1 per valid information (orientation, where on the image, objects around, near), max 3 points 
- Attributes: +0.5 per valid attribute (color, material, observable property), max 2 points
-if he describe the wrong objetcts give an identification score of 0

Planning (B /5)
- Actions (3 pts): Approach 0.5pts, Pre-grasp 0.75pts, Grasp 1.0pts, Lift 0.75pts
- Quality (2 pts): logical order 1, efficiency 1

Use (C /5)
- Realism (2 pts): number of valid uses ≥2=2, 1 valid=1
- Link Grasp–Use (2 pts): explicit=2, implicit=1, absent=0
- Physics (1 pt): consider weight, balance, fragility

Grasp (D /5)
- Type (2 pts): explicit correct=2, implicit=1, incorrect=0
- Contact zone (2 pts): precise=2, approximate=1, absent=0
- Accessibility (1 pt): consider orientation, obstacles, reachability

Robustness (E /5, rescaled)
- VVL Hallucination (1 pt): none=1, minor=0.5, major/invalid=0
- Logic (1 pt): perfect=1, minor=0.5, incoherent=0
- Rescaled E = raw × 2.5

Final Score = (A + B + C + D + E) /5

VLM Response:
\"\"\"
{vlm_response}
\"\"\"

Provide the output **strictly as JSON**:

{{
  "identification": {{"position_score": ..., "attribute_score": ..., "total": ...}},
  "planning": {{"actions_score": ..., "quality_score": ..., "total": ...}},
  "use": {{"realism_score": ..., "link_score": ..., "physics_score": ..., "total": ...}},
  "grasp": {{"type_score": ..., "zone_score": ..., "access_score": ..., "total": ...}},
  "robustness": {{"vvl_score": ..., "logic_score": ..., "total_raw": ..., "total_scaled": ...}},
  "final_score": ...
}}
"""

    content = [{"type": "text", "text": prompt}]
    for img_b64 in images_base64:
        content.append({"type": "image_url", "image_url": {"url": img_b64}})

    response = or_client.chat.send(
        model=model_id,
        messages=[{"role": "user", "content": content}],
        stream=False,
        temperature=0,
        top_p=1,
        seed=42
    )

    raw_output = response.choices[0].message.content
    return safe_json_parse(raw_output)

# MAIN

with open("test2gen.json", "r") as f:
    vlm_data = json.load(f)

results = []

for entry in vlm_data:
    obj_name = entry["object"]
    vlm_response = json.dumps(entry["response"], indent=2)
    model_name_readable = entry.get("model", "Unknown")
    
    print(f"Evaluating {model_name_readable} response for {obj_name}...")
    start = time.time()
    try:
        score_json = evaluate_response(obj_name, vlm_response, model_id="openai/gpt-5.2")

        if "error" in score_json:
            score_json = {
                "error": score_json.get("error"),
                "raw": score_json.get("raw"),
                "final_score": None
            }
    except Exception as e:
        score_json = {"error": str(e)}
    end = time.time()

    score_json["object"] = obj_name
    score_json["vlm_model"] = model_name_readable
    score_json["response_time"] = end - start

    results.append(score_json)

# SAVE RESULTS

with open("test2evalchat.json", "w") as f:
    json.dump(results, f, indent=2)

df = pd.json_normalize(results)
df.to_csv("test2evalchat.csv", index=False)

print("\n✅ Evaluation complete. Results saved to JSON + CSV")