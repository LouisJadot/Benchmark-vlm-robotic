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
OUTPUT_JSON = "test5gen.json"
OUTPUT_CSV = "test5gen.csv"

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
            "tidy the desk by clearing the keyboard area, removing loose items like the cup and mouse from the keyboard, organizing cables, and preparing the computer setup for use by turning on the tower and monitor"
        ]
    },
    "office_disorder2": {
        "images": ["benchmarkvlm/data/datatest4,5/desk.jpeg"],
        "task_sequence": [
            "tidy the desk"
        ]
    },
        "office_disorder3": {
        "images": ["benchmarkvlm/data/datatest4,5/desk.jpeg"],
        "task_sequence": [
            "turn on the computer "
        ]
    },
    "office_disorder4": {
        "images": ["benchmarkvlm/data/datatest4,5/desk.jpeg"],
        "task_sequence": [
            "set up the computer for use"
        ]
    },
    "office_disorder5": {
        "images": ["benchmarkvlm/data/datatest4,5/desk.jpeg"],
        "task_sequence": [
            "Adjust the webcam"
        ]
    },

    "table_after_meal_01": {
        "images": ["benchmarkvlm/data/datatest4,5/table.jpeg"],
        "task_sequence": [
            "clean and reset the table after a meal by stacking the glasses, placing cutlery on plates, and collecting plates and glassware for removal"
        ]
    },

    "table_after_meal_02": {
        "images": ["benchmarkvlm/data/datatest4,5/table.jpeg"],
        "task_sequence": [
            "clean and reset the table"
        ]
    },
    "table_after_meal_03": {
        "images": ["benchmarkvlm/data/datatest4,5/table.jpeg"],
        "task_sequence": [
            "clean the table"
        ]
    },
    "table_after_meal_04": {
        "images": ["benchmarkvlm/data/datatest4,5/table.jpeg"],
        "task_sequence": [
            "tidy the table"
        ]
    },

    "table_after_meal_05": {
        "images": ["benchmarkvlm/data/datatest4,5/table.jpeg"],
        "task_sequence": [
            "stacks the cutlery and glasses on the plate "
        ]
    },

    "living_room": {
        "images": ["benchmarkvlm/data/datatest4,5/livingroom.jpeg"],
        "task_sequence": [
            "organize the living room by stacking scattered objects, placing items like pens and cables in proper locations, closing the computer, and arranging cushions neatly on the sofa"
        ]
    },

    "living_room2": {
        "images": ["benchmarkvlm/data/datatest4,5/livingroom.jpeg"],
        "task_sequence": [
            "organize the living room"
        ]
    },

    "living_room3": {
        "images": ["benchmarkvlm/data/datatest4,5/livingroom.jpeg"],
        "task_sequence": [
            "organize the sofa"
        ]
    },

    "living_room4": {
        "images": ["benchmarkvlm/data/datatest4,5/livingroom.jpeg"],
        "task_sequence": [
            "organize the coffee table"
        ]
    },
    "living_room5": {
        "images": ["benchmarkvlm/data/datatest4,5/livingroom.jpeg"],
        "task_sequence": [
            "arrange the cushions on the sofa "
        ]
    },

    "room": {
        "images": ["benchmarkvlm/data/datatest4,5/room.jpeg"],
        "task_sequence": [
            "sort and store items by hanging the jacket, organizing shoes, disposing of waste, and placing clothes and accessories into appropriate storage bags"
        ]
    },
    "room2": {
        "images": ["benchmarkvlm/data/datatest4,5/room.jpeg"],
        "task_sequence": [
            "tidy the room"
        ]
    },
    "room3": {
        "images": ["benchmarkvlm/data/datatest4,5/room.jpeg"],
        "task_sequence": [
            "put the items lying around on the bed and wants to get on the bed,"
        ]
    },
    "room4": {
        "images": ["benchmarkvlm/data/datatest4,5/room.jpeg"],
        "task_sequence": [
            "put the clothes lying around in the laundry bag"
        ]
    },
    "room5": {
        "images": ["benchmarkvlm/data/datatest4,5/room.jpeg"],
        "task_sequence": [
            "put the jacket away with the other clothes"
        ]
    },

    "kitchen": {
        "images": ["benchmarkvlm/data/datatest4,5/kitchen.jpeg"],
        "task_sequence": [
            "organize the kitchen"
        ]
    },
    "kitchen2": {
        "images": ["benchmarkvlm/data/datatest4,5/kitchen.jpeg"],
        "task_sequence": [
            "tidy the dishes"
        ]
    },
    "kitchen3": {
        "images": ["benchmarkvlm/data/datatest4,5/kitchen.jpeg"],
        "task_sequence": [
            "put the dishes away in the cupboards"
        ]
    },
    "kitchen4": {
        "images": ["benchmarkvlm/data/datatest4,5/kitchen.jpeg"],
        "task_sequence": [
            "put the dishes in the dishwasher "
        ]
    },
    "kitchen5": {
        "images": ["benchmarkvlm/data/datatest4,5/kitchen.jpeg"],
        "task_sequence": [
            "grab the biggest pot"
        ]
    },

    "bathroom": {
        "images": ["benchmarkvlm/data/datatest4,5/bathroom.jpeg"],
        "task_sequence": [
            "clean and organize the bathroom by placing toiletries in proper locations, disposing of empty items, arranging towels, and closing the shower curtain"
        ]
    },
    "bathroom2": {
        "images": ["benchmarkvlm/data/datatest4,5/bathroom.jpeg"],
        "task_sequence": [
            "clean the bathroom "
        ]
    },
    "bathroom3": {
        "images": ["benchmarkvlm/data/datatest4,5/bathroom.jpeg"],
        "task_sequence": [
            "organize the bathroom "
        ]
    },
    "bathroom4": {
        "images": ["benchmarkvlm/data/datatest4,5/bathroom.jpeg"],
        "task_sequence": [
            "get the shower ready so it's ready to use "
        ]
    },
    "bathroom5": {
        "images": ["benchmarkvlm/data/datatest4,5/bathroom.jpeg"],
        "task_sequence": [
            "Turn off the water and close the shower curtain "
        ]
    },

    "sink": {
        "images": ["benchmarkvlm/data/datatest4,5/sink.jpeg"],
        "task_sequence": [
            "clean and reset the sink area by turning on the tap, placing cutlery in the sink, organizing cleaning tools, and turning off the tap"
        ]
    },
    "sink2": {
        "images": ["benchmarkvlm/data/datatest4,5/sink.jpeg"],
        "task_sequence": [
            "clean the sink area"
        ]
    },
    "sink3": {
        "images": ["benchmarkvlm/data/datatest4,5/sink.jpeg"],
        "task_sequence": [
            "does the dishes"
        ]
    },
    "sink4": {
        "images": ["benchmarkvlm/data/datatest4,5/sink.jpeg"],
        "task_sequence": [
            "washes the plates and cutlery"
        ]
    },
    "sink5": {
        "images": ["benchmarkvlm/data/datatest4,5/sink.jpeg"],
        "task_sequence": [
            "washes the pot and pan "
        ]
    },

    "cupboard": {
        "images": ["benchmarkvlm/data/datatest4,5/cupboard.jpeg"],
        "task_sequence": [
            "organize the storage cabinet by stacking bowls and plates correctly and closing the cupboard"
        ]
    },
    "cupboard2": {
        "images": ["benchmarkvlm/data/datatest4,5/cupboard.jpeg"],
        "task_sequence": [
            "organize the cupboard"
        ]
    },
    "cupboard3": {
        "images": ["benchmarkvlm/data/datatest4,5/cupboard.jpeg"],
        "task_sequence": [
            "organize the plates"
        ]
    },
    "cupboard4": {
        "images": ["benchmarkvlm/data/datatest4,5/cupboard.jpeg"],
        "task_sequence": [
            "organize the bowls"
        ]
    },
    "cupboard5": {
        "images": ["benchmarkvlm/data/datatest4,5/cupboard.jpeg"],
        "task_sequence": [
            "reorganize the cupboard"
        ]
    },
    "bathroomsink": {
        "images": ["benchmarkvlm/data/datatest4,5/bathroomsink.jpeg"],
        "task_sequence": [
            "organize the bathroom sink area by repositioning toiletries, placing items in logical spatial order, arranging the mat, and turning on the tap",
        ]
    },
    "bathroomsink2": {
        "images": ["benchmarkvlm/data/datatest4,5/bathroomsink.jpeg"],
        "task_sequence": [
            "get the bathroom ready so it's ready to use"
        ]
    },
    "bathroomsink3": {
        "images": ["benchmarkvlm/data/datatest4,5/bathroomsink.jpeg"],
        "task_sequence": [
            "organize the bathroom sink area "
        ]
    },
    "bathroomsink4": {
        "images": ["benchmarkvlm/data/datatest4,5/bathroomsink.jpeg"],
        "task_sequence": [
            "Rinse the toothbrush"
        ]
    },
    "bathroomsink5": {
        "images": ["benchmarkvlm/data/datatest4,5/bathroomsink.jpeg"],
        "task_sequence": [
            "Rinse the toothbrush and put it away"
        ]
    }
    
}

# PROMPT


prompt_template = """
You are a robotics perception and manipulation system.

You are given:
- an image of a real-world scene
- an ordered task (may be high-level)

Your goal is to produce a grounded, physically consistent execution plan.

━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES (STRICT)
━━━━━━━━━━━━━━━━━━━━━━

1. Only use objects visible in the image.
2. Do NOT hallucinate tools or hidden objects.
3. All actions must be physically executable.
4. Maintain consistent object naming.
5. NO abstract reasoning in final plan (only grounded physical transformations).
6. Output MUST be valid JSON ONLY.

━━━━━━━━━━━━━━━━━━━━━━
HIERARCHICAL PLANNING RULE (STRICT HTN STYLE)
━━━━━━━━━━━━━━━━━━━━━━

YOU MUST FOLLOW 2 PHASES:

PHASE 1 — SUBTASK DECOMPOSITION (NO ACTIONS)
- Convert the high-level task into atomic state transitions
- Each subtask MUST describe a single and CLEARphysical transformation

PHASE 2 — EXECUTION PLAN
- Each plan step MUST correspond to exactly ONE subtask
- No new reasoning allowed in plan
- No new objects or goals allowed in plan

━━━━━━━━━━━━━━━━━━━━━━
FORBIDDEN SUBTASK WORDS
━━━━━━━━━━━━━━━━━━━━━━

DO NOT USE:
- clean
- tidy
- organize
- prepare
- fix
- arrange (alone)

━━━━━━━━━━━━━━━━━━━━━━
SUBTASK GENERATION RULE
━━━━━━━━━━━━━━━━━━━━━━

Each subtask MUST:

1. Describe ONE atomic state change
2. Include:
   - source location
   - target location
   - object(s)
3. Be executable in 1–3 robot actions max
4. Fully cover the task

━━━━━━━━━━━━━━━━━━━━━━
TASK TO DECOMPOSE
━━━━━━━━━━━━━━━━━━━━━━

{tasks}

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT JSON)
━━━━━━━━━━━━━━━━━━━━━━

{{
  "scene_understanding": {{
    "visible_objects": ["..."],
    "spatial_layout": "..."
  }},

  "task_decomposition": {{
    "high_level_goal": "...",

    "subtasks": [
      {{
        "id": 1,
        "action": "...",
        "goal": "explicit physical transformation",
        "target_objects": ["..."],

        "from_state": "location before",
        "to_state": "location after",

        "required_effect": "observable change in scene",
        "constraints": "physical constraints"
      }}
    ]
  }},

  "plan": [
    {{
      "step": 1,
      "subtask_id": 1,

      "object": {{
        "name": "...",
        "target_objects": ["..."],
        "required_effect": "...",
        "constraints": "...",
        "attributes": "...",

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

      "precondition": "...",
      "postcondition": "...",

      "state_after": "..."
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
                print(f"⚠️ Missing image: {img_path}")

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

print("\ DONE")