from openai import OpenAI
import base64
import time

client = OpenAI()

prompt_text = """
You are given a photo with an object.
Tasks:
1. Identify the object
2. Explain how a robot could grasp it
3. Provide step-by-step grasp plan
4. Suggest uses linked to grasp
"""

image_path = "data/wetransfer_image00001-jpeg_2026-03-30_2141/image00014.jpeg"

# Convertir l'image en base64
with open(image_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

start = time.time()

response = client.responses.create(
    model="gpt-5.4-nano",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}", "detail": "original"}
            ]
        }
    ]
)

end = time.time()

print("Output:\n", response.output_text)
print("Time:", end-start, "seconds")