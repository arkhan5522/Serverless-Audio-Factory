# generator.py
import os
import subprocess
import sys

# ==========================================
# 1. INSTALL DEPENDENCIES
# ==========================================
print("Installing Chatterbox TTS...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chatterbox-tts", "--quiet"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchaudio", "--quiet"])
except Exception as e:
    print(f"Failed to install dependencies: {e}")
    # We continue because sometimes they are already installed in cached kernels

# ==========================================
# 2. IMPORT LIBRARIES
# ==========================================
import torch
import torchaudio as ta
import requests
from chatterbox.tts import ChatterboxTTS

# --- INJECTED_VARIABLES_START ---
TEXT_TO_SPEAK = """{{TEXT_PLACEHOLDER}}"""
AUDIO_URL = """{{AUDIO_URL_PLACEHOLDER}}"""
# --- INJECTED_VARIABLES_END ---

print("Setting up...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 3. Download Reference Audio
ref_audio_path = "reference_voice.mp3"
print(f"Attempting to download from: {AUDIO_URL}")

try:
    token = os.environ.get('GH_PAT', '')
    
    # DEBUG: Check if token exists (printing just the length for security)
    if len(token) > 0:
        print(f"Token detected (Length: {len(token)}). Auth should work.")
    else:
        print("WARNING: No Token detected. Download will fail if repo is private.")

    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    # CRITICAL: This header tells GitHub API to give us the Raw File Bytes
    headers['Accept'] = 'application/vnd.github.v3.raw'
    
    # We download the file
    r = requests.get(AUDIO_URL, headers=headers, allow_redirects=True)
    
    if r.status_code == 200:
        with open(ref_audio_path, "wb") as f:
            f.write(r.content)
        print("Download successful.")
    else:
        print(f"Download failed with Code: {r.status_code}")
        print(f"Response: {r.text[:300]}") # Print error details
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
