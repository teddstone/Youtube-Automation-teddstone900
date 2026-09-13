import os
import io
from PIL import Image
from huggingface_hub import InferenceClient

# Initialize our free serverless client via your secure GitHub Secret token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Error: HF_TOKEN environment variable is missing!")

client = InferenceClient(token=hf_token)

def generate_automation_content():
    topic = "Space exploration facts"
    print(f"--- Starting Automation Pipeline for topic: {topic} ---")
    
    # 1. Generate text script using chat_completion (Fixes the previous error)
    print("Generating script...")
    chat_completion = client.chat_completion(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "user", "content": f"Write a 1-sentence interesting fact about: {topic}."}
        ],
        max_tokens=100
    )
    script_output = chat_completion.choices.message.content.strip()
    print(f"Generated Script: {script_output}\n")
    
    # 2. Convert that text script into an image prompt
    image_prompt = f"A dramatic cinematic digital art depiction of: {script_output}. 4k, photorealistic."
    print(f"Generating image frame with prompt: {image_prompt}...")
    
    # 3. Request visual generation via FLUX
    image_bytes = client.text_to_image(
        prompt=image_prompt,
        model="black-forest-labs/FLUX.1-schnell"
    )
    
    # 4. Save file to cloud workspace
    image = Image.open(io.BytesIO(image_bytes))
    output_path = "output_video_frame.png"
    image.save(output_path)
    print(f"Successfully saved AI visual asset to: {output_path}")
    print("--- Pipeline Testing Step Complete ---")

if __name__ == "__main__":
    generate_automation_content()
