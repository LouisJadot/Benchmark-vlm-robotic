from openrouter import OpenRouter
import pandas as pd
import time
import os
import base64
import json

# CLIENT


or_client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

# IMAGE (BASE64)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

# SAFE JSON PARSE


def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {"error": "invalid_json", "raw": text}

# DATASET IMG

object_images = {
    "blue pen": ["benchmarkvlm/data/datatest1,2,3/bluecan1.jpeg", "benchmarkvlm/data/datatest1,2,3/bluepen2.jpeg"],
    "grey mug": ["benchmarkvlm/data/datatest1,2,3/greymug1.jpeg", "benchmarkvlm/data/datatest1,2,3/greymug2.jpeg"],
    "computer mouse": ["benchmarkvlm/data/datatest1,2,3/computermouse1.jpeg", "benchmarkvlm/data/datatest1,2,3/computermouse2.jpeg","benchmarkvlm/data/datatest1,2,3/computermouse3.jpeg"],
    "orange balloon": ["benchmarkvlm/data/datatest1,2,3/orangeballoon1.jpeg", "benchmarkvlm/data/datatest1,2,3/orangeballoon2.jpeg"],
    "blue can": ["benchmarkvlm/data/datatest1,2,3/bluecan1.jpeg", "benchmarkvlm/data/datatest1,2,3/bluecan2.jpeg"],
    "USB drive": ["benchmarkvlm/data/datatest1,2,3/usbdrive1.jpeg", "benchmarkvlm/data/datatest1,2,3/usbdrive2.jpeg"],
    "whiteboard eraser": ["benchmarkvlm/data/datatest1,2,3/whiteboarderaser1.jpeg", "benchmarkvlm/data/datatest1,2,3/whiteboarderaser2.jpeg"],
    "stirrer": ["benchmarkvlm/data/datatest1,2,3/stirrer1.jpeg", "benchmarkvlm/data/datatest1,2,3/stirrer2.jpeg"]
}

# PROMPT JSON 

prompt_template = """
You are a robotics perception and manipulation expert.

You are given one or multiple images of an object.

STRICT RULES:
- Output MUST be valid JSON
- No markdown
- No comments
- No trailing commas
- No extra text
- All fields MUST be present
- All values MUST be strings (NO nested objects except "use" list)

TASKS:
1. Identify the object
2. Describe how a robot grasps it
3. Provide grasp steps
4. Suggest 2 uses

Object hint: {}

OUTPUT JSON FORMAT:

{{
  "object_identification": {{
    "name": "...",
    "attributes": "..."
  }},
  "grasp": {{
    "type": "...",
    "contact_zone": "...",
    "accessibility": "..."
  }},
  "planning": {{
    "approach": "...",
    "pre_grasp_orientation": "...",
    "grasp_contact": "...",
    "lift_stabilization": "..."
  }},
  "use": [
    {{
      "description": "...",
      "link_to_grasp": "...",
      "physical_constraints": "..."
    }},
    {{
      "description": "...",
      "link_to_grasp": "...",
      "physical_constraints": "..."
    }}
  ],
  "robustness": {{
    "assumptions": "...",
    "uncertainty_handling": "..."
  }}
}}
"""


# APPEL MODELE (MULTIMODAL)

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
                print(f" Image not found: {img_path}")

    response = or_client.chat.send(
        model=model_name,
        messages=[{
            "role": "user",
            "content": content
        }],
        stream=False,
        temperature=0.2,
        top_p=1,
        seed=42
    )

    raw_output = response.choices[0].message.content

    return safe_json_parse(raw_output)

# MODELS À TESTER

models = {
    "GPT-5.4 Mini": "openai/gpt-5.4-mini",
    "Qwen 3.5 9B": "qwen/qwen3.5-9b",
    "Gemini 3.1 Flash Image": "google/gemini-3.1-flash-image-preview",
    "Ministral 8B": "mistralai/ministral-8b-2512",
    "Seed 2.0 Lite": "bytedance-seed/seed-2.0-lite"
}

# MAIN LOOP

results = []

for obj, images in object_images.items():
    print(f"\n🔍 Object: {obj}")

    for model_name, model_id in models.items():
        prompt = prompt_template.format(obj)

        start = time.time()
        try:
            output = call_model(model_id, prompt, images)
        except Exception as e:
            output = {"error": str(e)}
        end = time.time()

        response_time = end - start
        print(f"{model_name} → {response_time:.2f}s")

        results.append({
            "object": obj,
            "model": model_name,
            "model_id": model_id,
            "images": ", ".join(images),
            "response": output,
            "response_time": response_time
        })

# SAVE RESULTS

df = pd.DataFrame(results)
df.to_csv("test1genrun1.csv", index=False)

with open("test1genrun1.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDONE — results saved to CSV + JSON")