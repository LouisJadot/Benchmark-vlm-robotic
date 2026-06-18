from openrouter import OpenRouter
import pandas as pd
import time
import os
import base64
import json

# CONFIG

if os.getenv("OPENROUTER_API_KEY") is None:
    raise ValueError("❌ OPENROUTER_API_KEY not set")

or_client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

N_RUNS = 3
OUTPUT_JSON = "test4gen.json"
OUTPUT_CSV = "test4gen.csv"

# ENCODE IMAGE

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

# ROBUST JSON PARSE 

def parse_json_strict(text):
    try:
        return json.loads(text), True
    except:
        try:
            # enlève ```json ``` si présent
            text = text.replace("```json", "").replace("```", "")
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end]), True
        except Exception as e:
            return {
                "error": "invalid_json",
                "raw_output": text,
                "exception": str(e)
            }, False


# DATASET

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

# PROMPT

prompt_template = """
You are a robotics perception and manipulation system.

You are given:
- an image of a real-world scene
- an ordered list of manipulation tasks

Your goal is to produce a grounded, physically consistent execution plan.

━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES (STRICT)
━━━━━━━━━━━━━━━━━━━━━━

1. Only use objects that are visually present in the image.
   - Do NOT invent tools (no robotic arm, no gripper, no hand unless visible)
   - Do NOT hallucinate hidden objects

2. Each step MUST be physically executable by a real robot.

3. Each step MUST produce a clear and verifiable state change.

4. Object naming MUST be consistent across all steps.

5. Each step MUST be self-contained:
   - include object properties
   - include grasp strategy
   - include action and resulting state

6. Output MUST be valid JSON ONLY
   - no markdown
   - no comments
   - no extra text

━━━━━━━━━━━━━━━━━━━━━━
ACTION TYPES (CHOOSE EXACTLY ONE)
━━━━━━━━━━━━━━━━━━━━━━

pick_and_place
push
pull
rotate
flip
press
open
close
lift
set_down
switch_on
switch_off
adjust
hold

━━━━━━━━━━━━━━━━━━━━━━
GRASP TYPES (CHOOSE EXACTLY ONE)
━━━━━━━━━━━━━━━━━━━━━━

precision_grasp
power_grasp
pinch_grasp
lateral_grasp
two_finger_grasp
whole_hand_grasp
support_grasp

━━━━━━━━━━━━━━━━━━━━━━
STATE MODELING RULE
━━━━━━━━━━━━━━━━━━━━━━

"state_after" MUST describe:
- object positions
- spatial relations (on, in, next_to, away_from, etc.)
- observable effects of the action

It must be directly verifiable.

━━━━━━━━━━━━━━━━━━━━━━
TASK SEQUENCE
━━━━━━━━━━━━━━━━━━━━━━

{tasks}

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT JSON)
━━━━━━━━━━━━━━━━━━━━━━

{{
  "scene_understanding": {{
    "visible_objects": ["..."],
    "spatial_layout": "short description of scene layout"
  }},

  "plan": [
    {{
      "step": 1,
      "task": "...",

      "object": {{
        "name": "...",
        "attributes": "visual + functional properties",

        "grasp": {{
          "type": "...",
          "contact_zone": "...",
          "accessibility": "easy | medium | hard"
        }},

        "physical_properties": {{
          "stability": "...",
          "fragility": "...",
          "manipulability": "low | medium | high"
        }}
      }},

      "action": "...",
      "action_type": "...",

      "precondition": "what must be true before action",
      "postcondition": "verifiable result after action",

      "state_after": "final scene state after action"
    }}
  ]
}}
"""

# CALL MODEL 

def call_model(model_name, prompt, images=None):

    content = [{"type": "text", "text": prompt}]

    if images:
        for img_path in images:
            if os.path.exists(img_path):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image(img_path)
                    }
                })
            else:
                print(f"Missing image: {img_path}")

    response = or_client.chat.send(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        stream=False,
        temperature=0.1,
        top_p = 1,
        seed = 42
    )

    raw_output = response.choices[0].message.content

    parsed, is_valid = parse_json_strict(raw_output)

    return parsed, is_valid, raw_output

# MODELS

models = {
    "GPT-5.4 Mini": "openai/gpt-5.4-mini",
    "Qwen 3.5 9B": "qwen/qwen3.5-9b",
    "Gemini Flash Image": "google/gemini-3.1-flash-image-preview",
    "Ministral 8B": "mistralai/ministral-8b-2512",
    "Seed 2.0 Lite": "bytedance-seed/seed-2.0-lite"
}

# MAIN LOOP

results = []

for scene_id, scene_data in scenes.items():
    print(f"\n🎬 Scene: {scene_id}")

    formatted_tasks = "\n".join(
        [f"{i+1}. {t}" for i, t in enumerate(scene_data["task_sequence"])]
    )

    prompt = prompt_template.format(tasks=formatted_tasks)

    for model_name, model_id in models.items():
        for run_id in range(N_RUNS):

            start = time.time()

            try:
                parsed, is_valid, raw_output = call_model(
                    model_id, prompt, scene_data["images"]
                )
                error = None
            except Exception as e:
                parsed = {"error": str(e)}
                is_valid = False
                raw_output = ""
                error = str(e)

            end = time.time()

            print(f"{model_name} run {run_id} → {end-start:.2f}s")

            results.append({
                "scene": scene_id,
                "model": model_name,
                "run_id": run_id,
                "response": parsed,
                "is_valid_json": is_valid,
                "error": error,
                "response_time": end - start
            })

# SAVE

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print("\n✅ DONE")