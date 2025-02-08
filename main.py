import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy

# Load spaCy for sentence splitting
nlp = spacy.load("en_core_web_sm")

# Model name
MODEL_NAME = "deepseek-ai/deepseek-chat-1.3b"

def load_model():
    """Check if model exists, otherwise download and load it."""
    try:
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
        print("Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have internet access for first-time setup.")
        exit(1)

# Load model & tokenizer
tokenizer, model = load_model()

def generate_scene_description(sentence):
    """Generate a 3-4 word movie-style scene description using DeepSeek."""
    prompt = f"Describe this scene in 3-4 words like a movie director: {sentence}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.7)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part
    scene_description = generated_text.replace(prompt, "").strip()
    
    return scene_description

def process_text(text):
    """Split text into sentences and generate scene descriptions."""
    doc = nlp(text)
    scene_descriptions = []
    
    for i, sent in enumerate(doc.sents, 1):
        description = generate_scene_description(sent.text)
        scene_descriptions.append(f"Scene {i}: {description}")

    return scene_descriptions

# Example input text
text = """Imagine waking up one day and realizing that your body is no longer just yours. 
Every heartbeat, every breath, every meal—it’s now shared with another life. 
Pregnancy doesn’t just change a person for nine months. It changes them forever. 
The heart grows bigger to pump 50% more blood. 
Ligaments stretch, bones shift, and the brain actually shrinks—but in a good way! 
The immune system recalibrates itself to protect both mother and child. 
Blood sugar levels rise, the body stores more fat, and the respiratory system expands. 
Hormones surge like a rollercoaster, leading to mood swings. 
Fetal cells stay in the mother’s body for decades, influencing her immune system. 
Pregnancy is about transformation in ways we’re only beginning to understand."""

# Run the model
scenes = process_text(text)

# Print results
for scene in scenes:
    print(scene)
