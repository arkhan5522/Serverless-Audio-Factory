# generator.py
import os
import subprocess
import sys

# ==========================================
# 1. INSTALL DEPENDENCIES (CRITICAL STEP)
# ==========================================
print("Installing Chatterbox TTS... (This may take a minute)")
try:
    # We use subprocess to run pip inside the Kaggle script
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chatterbox-tts", "--quiet"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchaudio", "--quiet"])
except Exception as e:
    print(f"Failed to install dependencies: {e}")
    exit(1)

# ==========================================
# 2. NOW IMPORT LIBRARIES
# ==========================================
# We only import these AFTER the installation above succeeds
import torch
import torchaudio as ta
import requests
# The critical import that was failing:
from chatterbox.tts import ChatterboxTTS

# --- INJECTED_VARIABLES_START ---
TEXT_TO_SPEAK = """{{TEXT_PLACEHOLDER}}"""
AUDIO_URL = """{{AUDIO_URL_PLACEHOLDER}}"""
# --- INJECTED_VARIABLES_END ---

print("Setting up...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 3. Download Reference Audio (Authenticated)
ref_audio_path = "reference_voice.mp3"
print(f"Downloading audio from: {AUDIO_URL}")

try:
    # Get the token from the environment variable we passed in the .yml
    token = os.environ.get('GH_PAT', '')
    headers = {}
    
    # Only add headers if token exists (for private repos)
    if token:
        headers['Authorization'] = f'token {token}'
        headers['Accept'] = 'application/vnd.github.v3.raw'
    
    # We need to handle potential redirects for raw content
    r = requests.get(AUDIO_URL, headers=headers, allow_redirects=True)
    
    if r.status_code == 200:
        with open(ref_audio_path, "wb") as f:
            f.write(r.content)
        print("Download successful.")
    else:
        print(f"Download failed: {r.status_code}")
        # Print first 100 chars of response to debug
        print(r.text[:200]) 
        exit(1)
        
except Exception as e:
    print(f"Error during download: {e}")
    exit(1)

# 4. Load Model
print("Loading Chatterbox model...")
try:
    model = ChatterboxTTS.from_pretrained(device=device)
except Exception as e:
    print(f"Model Load Error: {e}")
    exit(1)

# 5. Generate Speech
print("Synthesizing speech...")
try:
    with torch.no_grad():
        wav = model.generate(
            text=TEXT_TO_SPEAK,
            audio_prompt_path=ref_audio_path,
            exaggeration=0.5,
            cfg_weight=0.5
        )

    # 6. Save Output
    output_filename = "output.wav"
    ta.save(output_filename, wav.cpu(), model.sr)
    print("Audio saved successfully.")

except Exception as e:
    print(f"Generation Error: {e}")
    exit(1)
