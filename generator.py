# generator.py
import os
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import requests

# --- INJECTED_VARIABLES_START ---
TEXT_TO_SPEAK = """{{TEXT_PLACEHOLDER}}"""
AUDIO_URL = """{{AUDIO_URL_PLACEHOLDER}}"""
# --- INJECTED_VARIABLES_END ---

print("Setting up...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Download Reference Audio (Authenticated)
ref_audio_path = "reference_voice.mp3"
print(f"Downloading audio from: {AUDIO_URL}")

try:
    # Use the token we injected in the workflow step
    token = os.environ.get('GH_PAT', '')
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
        headers['Accept'] = 'application/vnd.github.v3.raw'
    
    r = requests.get(AUDIO_URL, headers=headers)
    
    if r.status_code == 200:
        with open(ref_audio_path, "wb") as f:
            f.write(r.content)
        print("Download successful.")
    else:
        print(f"Download failed: {r.status_code} {r.text}")
        exit(1)
        
except Exception as e:
    print(f"Error during download: {e}")
    exit(1)

# 2. Load Model & Audio
model = ChatterboxTTS.from_pretrained(device=device)

# 3. Generate
print("Generating speech...")
try:
    with torch.no_grad():
        wav = model.generate(
            text=TEXT_TO_SPEAK,
            audio_prompt_path=ref_audio_path,
            exaggeration=0.5,
            cfg_weight=0.5
        )

    # 4. Save Output
    output_filename = "output.wav"
    ta.save(output_filename, wav.cpu(), model.sr)
    print("Done.")

except Exception as e:
    print(f"Generation Error: {e}")
    exit(1)
