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

INPUT_JSON = "test5gen.json"
OUTPUT_JSON = "test5evalclaude.json"
OUTPUT_CSV = "test5evalclaude.csv"

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


# FINAL EVAL PROMPT 

eval_prompt = """
You are a STRICT robotics task-planning evaluation engine.

You evaluate a predicted hierarchical manipulation plan using ONLY:
- the input image
- the user task
- the predicted JSON

You MUST be:
- deterministic
- conservative
- objective
- literal

Never reward vague, generic, hallucinated, or physically impossible answers.

━━━━━━━━━━━━━━━━━━━━━━
CORE RULES
━━━━━━━━━━━━━━━━━━━━━━

1. Only visible objects are valid.
2. No hallucinated objects, tools, locations, or states.
3. Every statement must be grounded in the image or task.
4. If information is missing or unverifiable → lower the score.
5. No probabilistic interpretation.
6. Every criterion MUST appear explicitly in output.
7. Output STRICT JSON ONLY.
8. Judge semantic correctness, not exact wording.
9. Synonyms are acceptable if unambiguous.
10. A response must be evaluated for correctness and coherence, not only for field presence.

━━━━━━━━━━━━━━━━━━━━━━
ALLOWED SCORES
━━━━━━━━━━━━━━━━━━━━━━

ONLY these values are allowed:

- 0
- 0.5
- 1

Definitions:

0:
- incorrect
- hallucinated
- impossible
- contradictory
- missing
- not grounded in image/task

0.5:
- partially correct
- incomplete
- weakly grounded
- ambiguous but plausible
- minor inconsistency
- generic but still relevant

1:
- correct
- grounded
- coherent
- physically valid
- specific
- fully consistent

DO NOT use any other numeric value.

━━━━━━━━━━━━━━━━━━━━━━
INPUTS
━━━━━━━━━━━━━━━━━━━━━━

TASK:
{tasks}

PREDICTED PLAN:
{prediction}

━━━━━━━━━━━━━━━━━━━━━━
LEVELS OF EVALUATION
━━━━━━━━━━━━━━━━━━━━━━

You MUST evaluate:

1. SCENE UNDERSTANDING
2. TASK DECOMPOSITION
3. ACTION PLAN
4. GLOBAL CONSISTENCY

━━━━━━━━━━━━━━━━━━━━━━
1) SCENE UNDERSTANDING EVALUATION
━━━━━━━━━━━━━━━━━━━━━━

Evaluate:

──────────────────────────────
SU1 — visible_objects_correct
──────────────────────────────

1:
- all listed objects visible
- no hallucinated objects

0.5:
- mostly correct objects
- minor naming ambiguity
- one uncertain object

0:
- hallucinated objects
- several incorrect objects
- objects absent from image

──────────────────────────────
SU2 — spatial_layout_correct
──────────────────────────────

1:
- spatial relations consistent with image

0.5:
- partially correct layout
- coarse but acceptable relations

0:
- contradictory or incorrect layout

──────────────────────────────
SU3 — naming_consistency
──────────────────────────────

1:
- same objects consistently referenced

0.5:
- small inconsistencies or synonyms

0:
- contradictory naming
- object switching

━━━━━━━━━━━━━━━━━━━━━━
2) TASK DECOMPOSITION EVALUATION
━━━━━━━━━━━━━━━━━━━━━━

For EACH subtask evaluate:

──────────────────────────────
TD1 — subtask_relevant
──────────────────────────────

1:
- directly contributes to task

0.5:
- loosely relevant
- partially useful

0:
- irrelevant

──────────────────────────────
TD2 — subtask_grounded
──────────────────────────────

1:
- all referenced objects visible

0.5:
- mostly grounded
- partially ambiguous objects

0:
- hallucinated or absent objects

──────────────────────────────
TD3 — subtask_atomic
──────────────────────────────

1:
- one concrete manipulation goal

0.5:
- slightly broad but executable

0:
- abstract
- multiple unrelated goals

GOOD:
- move cup off keyboard
- press monitor button

BAD:
- organize workspace
- prepare desk

──────────────────────────────
TD4 — subtask_physically_valid
──────────────────────────────

1:
- physically executable

0.5:
- executable but underspecified

0:
- physically impossible

──────────────────────────────
TD5 — subtask_complete
──────────────────────────────

1:
- subtasks collectively cover full task

0.5:
- partially complete
- minor missing manipulation

0:
- major missing requirements

━━━━━━━━━━━━━━━━━━━━━━
3) ACTION PLAN EVALUATION
━━━━━━━━━━━━━━━━━━━━━━

For EACH plan step evaluate:

──────────────────────────────
AP1 — referenced_subtask_exists
──────────────────────────────

1:
- referenced subtask exists

0.5:
- partially matching reference

0:
- missing or invalid reference

──────────────────────────────
AP2 — action_matches_subtask
──────────────────────────────

1:
- action directly implements subtask

0.5:
- partially aligned

0:
- unrelated action

──────────────────────────────
AP3 — target_object_valid
──────────────────────────────

1:
- object visible and correctly identified

0.5:
- object plausible but ambiguous
- category approximately correct

0:
- hallucinated or incorrect object

──────────────────────────────
AP4 — grasp_valid
──────────────────────────────

1:
- grasp fully appropriate

0.5:
- plausible but generic

0:
- physically incoherent grasp

──────────────────────────────
AP5 — contact_zone_valid
──────────────────────────────

1:
- coherent contact region

0.5:
- partially precise

0:
- impossible or incoherent contact

──────────────────────────────
AP6 — accessibility_valid
──────────────────────────────

1:
- accessible in scene

0.5:
- accessibility uncertain

0:
- inaccessible or blocked

──────────────────────────────
AP7 — physical_properties_valid
──────────────────────────────

1:
- coherent physical properties

0.5:
- partially correct or generic

0:
- contradictory properties

──────────────────────────────
AP8 — action_physically_possible
──────────────────────────────

1:
- realistically executable

0.5:
- plausible but underspecified

0:
- impossible action

──────────────────────────────
AP9 — precondition_valid
──────────────────────────────

1:
- coherent and necessary

0.5:
- generic but acceptable

0:
- contradictory or invalid

──────────────────────────────
AP10 — postcondition_valid
──────────────────────────────

1:
- logically follows action

0.5:
- partially coherent

0:
- incoherent result

──────────────────────────────
AP11 — state_after_valid
──────────────────────────────

1:
- resulting state coherent

0.5:
- partially coherent

0:
- contradictory state

━━━━━━━━━━━━━━━━━━━━━━
4) GLOBAL CONSISTENCY EVALUATION
━━━━━━━━━━━━━━━━━━━━━━

──────────────────────────────
GC1 — temporal_order_valid
──────────────────────────────

1:
- action order fully coherent

0.5:
- mostly coherent
- minor ordering issue

0:
- impossible ordering

──────────────────────────────
GC2 — no_state_contradiction
──────────────────────────────

1:
- no contradictions

0.5:
- small inconsistency

0:
- major contradiction

──────────────────────────────
GC3 — object_state_tracking_valid
──────────────────────────────

1:
- object states tracked coherently

0.5:
- minor tracking ambiguity

0:
- incoherent tracking

──────────────────────────────
GC4 — no_unused_subtasks
──────────────────────────────

1:
- all subtasks used

0.5:
- one weakly used subtask

0:
- several unused subtasks

──────────────────────────────
GC5 — no_extra_actions
──────────────────────────────

1:
- no irrelevant actions

0.5:
- one weakly relevant action

0:
- several irrelevant actions

──────────────────────────────
GC6 — plan_achieves_task
──────────────────────────────

1:
- final plan satisfies task

0.5:
- partially satisfies task

0:
- fails task

━━━━━━━━━━━━━━━━━━━━━━
AGGREGATION
━━━━━━━━━━━━━━━━━━━━━━

SU_avg =
(SU1 + SU2 + SU3) / 3

TD_avg =
mean of all TD_subtask_score

AP_avg =
mean of all AP_step_score

GC_avg =
(GC1 + GC2 + GC3 + GC4 + GC5 + GC6) / 6

FINAL_SCORE =
(SU_avg + TD_avg + AP_avg + GC_avg) / 4

OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━

{{
  "scene_understanding": {{
    "SU1_visible_objects_correct": 0|0.5|1,
    "SU2_spatial_layout_correct": 0|0.5|1,
    "SU3_naming_consistency": 0|0.5|1,
    "SU_avg":  float
  }},

  "task_decomposition": [
    {{
      "subtask_id": 1,

      "TD1_subtask_relevant": 0|0.5|1,
      "TD2_subtask_grounded": 0|0.5|1,
      "TD3_subtask_atomic": 0|0.5|1,
      "TD4_subtask_physically_valid": 0|0.5|1,
      "TD5_subtask_complete": 0|0.5|1,

      "TD_subtask_score": float
    }}
  ],

  "action_plan": [
    {{
      "step": 1,

      "AP1_referenced_subtask_exists": 0|0.5|1,
      "AP2_action_matches_subtask": 0|0.5|1,
      "AP3_target_object_valid": 0|0.5|1,
      "AP4_grasp_valid": 0|0.5|1,
      "AP5_contact_zone_valid": 0|0.5|1,
      "AP6_accessibility_valid": 0|0.5|1,
      "AP7_physical_properties_valid": 0|0.5|1,
      "AP8_action_physically_possible": 0|0.5|1,
      "AP9_precondition_valid": 0|0.5|1,
      "AP10_postcondition_valid": 0|0.5|1,
      "AP11_state_after_valid": 0|0.5|1,

      "AP_step_score": float
    }}
  ],

  "global_consistency": {{
    "GC1_temporal_order_valid": 0|0.5|1,
    "GC2_no_state_contradiction": 0|0.5|1,
    "GC3_object_state_tracking_valid": 0|0.5|1,
    "GC4_no_unused_subtasks": 0|0.5|1,
    "GC5_no_extra_actions": 0|0.5|1,
    "GC6_plan_achieves_task": 0|0.5|1,

    "GC_avg":  float
  }},

  "final_aggregation": {{
    "SU_avg":  float,
    "TD_avg":  float,
    "AP_avg":  float,
    "GC_avg":  float,

    "FINAL_SCORE":  float
  }}
}}

━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━

- JSON ONLY
- NO markdown
- NO explanations
- NO comments
- ALL criteria mandatory
- ALL scores explicit
- ALL scores ∈ {{0, 0.5, 1}} except averages
- Evaluate semantic correctness, not formatting
- Missing information lowers score
- Hallucinations lower score
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

print("\nEVALUATION DONE")