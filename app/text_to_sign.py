import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import re
from spellchecker import SpellChecker
from gramformer import Gramformer

DATASET_DIR = os.path.join(os.path.dirname(__file__), "data", "asl_alphabet_train")

# Function to get images from dataset (Search by letter)
def get_image(letter):
    # Search by uppercase letter
    folder = os.path.join(DATASET_DIR, letter.upper())
    # Search by lowercase letter
    if not os.path.exists(folder):
        folder = os.path.join(DATASET_DIR, letter.lower())
    # if letter not found in dataset
    if not os.path.exists(folder):
        print(f"Letter '{letter}' not found in dataset.")
        return None
    # Filter image files only (png, jpg, jpeg)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"No image files found for letter '{letter}'.")
        return None
    # Choose a random image from the train images for letter
    file = random.choice(files)
    # Image Path
    img_path = os.path.join(folder, file)
    return Image.open(img_path).resize((100, 100))  # Resize to same size


# Cleaning text function 
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove anything that's not a-z or space
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Check for correct spelling only (No grammar correction)
    spell = SpellChecker()
    words = text.split()
    corrected_words = []
    for word in words:
        corrected = spell.correction(word)
        corrected_words.append(corrected if corrected else word)
    spell_corrected = " ".join(corrected_words)

    text = " ".join(corrected_words)
    
    gf = Gramformer(models=1, use_gpu=False)
    corrected_sentences = list(gf.correct(spell_corrected, max_candidates=1))
    if corrected_sentences:
        text = corrected_sentences[0]
    else:
        text = spell_corrected


    return text



# Function to convert text to a series of sign images
def text_to_sign_image(text):
    images = []
    # To check spaces between words
    for char in text:
        if char == " ":
            # Skip spaces (no image at all)
            continue
        else:
            img = get_image(char)
        if img:  
            images.append(img)
    # If the letter is not in the dataset
    if not images:
        print("No valid letters found in dataset.")
        return None
    
    # Concatenate and display the translated images
    widths = sum(img.width for img in images)
    height = images[0].height
    combined = Image.new("RGB", (widths, height), color="white")
    # Place images in the same order as the letters
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width

    return combined


# Generate Image
def generate_sign_image(text, enable_correction=True):
    if enable_correction:
        cleaned = clean_text(text)
    else:
        # Convert text to lower cases and cleaned it
        cleaned = re.sub(r'[^a-zA-Z]', '', text.lower())

    img = text_to_sign_image(cleaned)
    if img is None:
        return None, None

    img_name = re.sub(r'[^a-zA-Z0-9_-]', '_', cleaned.strip())
    if img_name == "":
        img_name = "translation"
    img_name = f"{img_name}.png"

    return img, img_name
