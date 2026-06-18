from openrouter import OpenRouter
import json
import pandas as pd
import time
import os
import base64

# CONFIG

if os.getenv("OPENROUTER_API_KEY") is None:
    raise ValueError("OPENROUTER_API_KEY not set")

or_client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

INPUT_JSON = "test4gen.json"
OUTPUT_JSON = "test4evalclaude.json"
OUTPUT_CSV = "test4evalclaude.csv"

JUDGE_MODEL = "anthropic/claude-sonnet-4.6"

# ENCODE IMAGE

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

# SCENES

scenes = {
    "office_disorder": {
        "images": ["benchmarkvlm/data/datatest4,5/desk.jpeg"],
        "task_sequence": [
            "move the cup so that it is no longer on the keyboard",
            "move the mouse so that it is no longer on the keyboard",
            "turn the keyboard over",
            "switch on the computer tower",
            "switch on the computer monitor"
        ]
    },
    "table_after_meal": {
        "images": ["benchmarkvlm/data/datatest4,5/table.jpeg"],
        "task_sequence": [
            "stack the glasses",
            "place the knife on the plate",
            "place the fork on the plate",
            "pick up the stack of glasses",
            "pick up the plate"
        ]
    },
    "living_room": {
        "images": ["benchmarkvlm/data/datatest4,5/livingroom.jpeg"],
        "task_sequence": [
            "stack the wooden discs on top of the others",
            "place the pen next to the computer",
            "close the computer",
            "put the cushions back on the nearest sofa",
            "pick up the cable"
        ]
    },
    "laundry_room": {
        "images": ["benchmarkvlm/data/datatest4,5/room.jpeg"],
        "task_sequence": [
            "hang the jacket on a hanger",
            "put the shoes next to the others",
            "throw the white bag in the bin",
            "put the T-shirt in the laundry bag",
            "put the hat in the rucksack"
        ]
    },
    "kitchen": {
        "images": ["benchmarkvlm/data/datatest4,5/kitchen.jpeg"],
        "task_sequence": [
            "put the plate on the worktop",
            "put the large saucepan on the worktop",
            "put the second saucepan inside the first one",
            "open the cupboard in the middle",
            "open the dishwasher"
        ]
    },
    "bathroom": {
        "images": ["benchmarkvlm/data/datatest4,5/bathroom.jpeg"],
        "task_sequence": [
            "put the mask in the corner",
            "put the black shower gel on a shelf",
            "throw away the white shower gel",
            "put the towel in the corner",
            "close the curtain"
        ]
    },
    "sink": {
        "images": ["benchmarkvlm/data/datatest4,5/sink.jpeg"],
        "task_sequence": [
            "turn on the tap",
            "place the cutlery in the small sink",
            "pick up the red brush",
            "put the red brush back on the right-hand side of the sink",
            "turn off the tap"
        ]
    },
    "cupboard": {
        "images": ["benchmarkvlm/data/datatest4,5/cupboard.jpeg"],
        "task_sequence": [
            "place the middle bowl on the smallest stack of bowls",
            "place the left-hand bowl in the same spot as before",
            "move the small black and white plates to the top right stack",
            "place the small black plates on top of the large black plates",
            "close the cupboard"
        ]
    },
    "bathroom_sink": {
        "images": ["benchmarkvlm/data/datatest4,5/bathroomsink.jpeg"],
        "task_sequence": [
            "put the toothbrush holder at the back of the sink",
            "put the shaving foam behind the perfume",
            "put the toothpaste to the right of the soap",
            "place the mat in front of the washbasin",
            "turn on the tap"
        ]
    }
}

# FINAL EVAL PROMPT

eval_prompt = """
You are a STRICT robotics perception and manipulation evaluation engine.

You evaluate a predicted plan based ONLY on:
- the input image(s)
- the task sequence
- the predicted JSON

You MUST be deterministic, strict, and conservative.

━━━━━━━━━━━━━━━━━━━━━━
CORE RULES
━━━━━━━━━━━━━━━━━━━━━━

- Only visible objects are valid
- No hallucinated objects/tools
- No assumptions beyond the image
- If uncertain → LOWER score
- Each field must be evaluated explicitly
- Scores ∈ [0,1] only

━━━━━━━━━━━━━━━━━━━━━━
INPUTS
━━━━━━━━━━━━━━━━━━━━━━

TASKS:
{tasks}

PREDICTED PLAN:
{prediction}

━━━━━━━━━━━━━━━━━━━━━━
STEP-LEVEL SCORING (ALL ∈ [0,1])
━━━━━━━━━━━━━━━━━━━━━━

For EACH step:

──────────────────────────────
1) OBJECT VALIDITY (OV)
──────────────────────────────

object_visible = 1 if object is clearly visible, else 0
object_class_correct = 1 if correct type, else 0
attributes_consistent = 1 if attributes match image, else 0

OV = (object_visible + object_class_correct + attributes_consistent) / 3

If object_visible = 0 → ALL scores for this step = 0

──────────────────────────────
2) GRASP DEFINITION (GD)
──────────────────────────────

grasp_type_valid = 1 if appropriate for object, else 0
contact_zone_valid = 1 if physically correct, else 0
accessibility_valid = 1 if consistent with scene, else 0

GD = (grasp_type_valid + contact_zone_valid + accessibility_valid) / 3

──────────────────────────────
3) PHYSICAL PROPERTIES (PPC)
──────────────────────────────

stability_valid = 1 if realistic, else 0
fragility_valid = 1 if plausible, else 0
manipulability_valid = 1 if coherent, else 0

PPC = (stability_valid + fragility_valid + manipulability_valid) / 3

──────────────────────────────
4) ACTION EXECUTION (AE)
──────────────────────────────

action_matches_task = 1 else 0
action_physically_possible = 1 else 0
action_targets_correct_object = 1 else 0

AE = (action_matches_task + action_physically_possible + action_targets_correct_object) / 3

──────────────────────────────
5) STATE TRANSITION (ST)
──────────────────────────────

precondition_valid = 1 else 0
postcondition_verifiable = 1 else 0
state_after_consistent = 1 else 0

ST = (precondition_valid + postcondition_verifiable + state_after_consistent) / 3

━━━━━━━━━━━━━━━━━━━━━━
6) SCENE SCORING
━━━━━━━━━━━━━━━━━━━━━━

visible_objects_correct = 1 else 0
no_hallucination = 1 else 0
layout_correct = 1 else 0

SU = (visible_objects_correct + no_hallucination + layout_correct) / 3
━━━━━━━━━━━━━━━━━━━━━━
7) GLOBAL SCORING
━━━━━━━━━━━━━━━━━━━━━━

no_object_switch = 1 else 0
no_contradiction = 1 else 0
temporal_order_correct = 1 else 0

PC = (no_object_switch + no_contradiction + temporal_order_correct) / 3

━━━━━━━━━━━━━━━━━━━━━━
AGGREGATION
━━━━━━━━━━━━━━━━━━━━━━

Let N = number of steps

OV_avg  = mean(OV)
GD_avg  = mean(GD)
PPC_avg = mean(PPC)
AE_avg  = mean(AE)
ST_avg  = mean(ST)

FINAL SCORE =
(OV_avg + GD_avg + PPC_avg + AE_avg + ST_avg + SU + PC) / 7

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT JSON ONLY)
━━━━━━━━━━━━━━━━━━━━━━

{{
  "steps": [
    {{
      "step": 1,
      "OV": 0.0,
      "GD": 0.0,
      "PPC": 0.0,
      "AE": 0.0,
      "ST": 0.0
    }}
  ],

  "scene_understanding_score": 0.0,
  "plan_consistency_score": 0.0,

  "averages": {{
    "OV": 0.0,
    "GD": 0.0,
    "PPC": 0.0,
    "AE": 0.0,
    "ST": 0.0
  }},

  "final_score": 0.0
}}

RULES:
- JSON ONLY
- NO text
- NO markdown
- Scores ∈ [0,1]
- Be strict
"""

# CALL JUDGE

def call_judge(prompt, images):

    content = [{"type": "text", "text": prompt}]

    for img_path in images:
        if os.path.exists(img_path):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": encode_image(img_path)
                }
            })
        else:
            print(f"⚠️ Missing image: {img_path}")

    response = or_client.chat.send(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        top_p=1,
        seed=42
    )

    return response.choices[0].message.content

# ROBUST JSON PARSER

def parse_json(text):
    try:
        return json.loads(text)
    except:
        try:
            text = text.replace("```json", "").replace("```", "")
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {"error": "invalid_json", "raw": text}

# LOAD DATA

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

results = []

# MAIN LOOP

for item in data:

    scene = item["scene"]
    prediction = item["response"]

    if scene not in scenes:
        continue

    scene_data = scenes[scene]

    formatted_tasks = "\n".join(
        [f"{i+1}. {t}" for i, t in enumerate(scene_data["task_sequence"])]
    )

    prompt = eval_prompt.format(
        tasks=formatted_tasks,
        prediction=json.dumps(prediction, indent=2)
    )

    start = time.time()

    try:
        raw_eval = call_judge(prompt, scene_data["images"])
        parsed_eval = parse_json(raw_eval)
        error = None
    except Exception as e:
        parsed_eval = {"error": str(e)}
        error = str(e)

    end = time.time()

    print(f"Eval → {scene} | {item['model']} | run {item['run_id']}")

    results.append({
        "scene": scene,
        "model": item["model"],
        "run_id": item["run_id"],
        "evaluation": parsed_eval,
        "eval_time": end - start,
        "error": error
    })

# SAVE

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print("\n✅ EVALUATION DONE")