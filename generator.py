# generator.py
import os
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import urllib.request

# --- INJECTED_VARIABLES_START ---
TEXT_TO_SPEAK = """{{TEXT_PLACEHOLDER}}"""
AUDIO_URL = """{{AUDIO_URL_PLACEHOLDER}}"""
# --- INJECTED_VARIABLES_END ---

print("Setting up...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Download Reference Audio from the URL
ref_audio_path = "reference_voice.mp3"
print(f"Downloading audio from: {AUDIO_URL}")
try:
    urllib.request.urlretrieve(AUDIO_URL, ref_audio_path)
except Exception as e:
    print(f"Failed to download audio: {e}")
    exit(1)

# 2. Load Model & Audio
model = ChatterboxTTS.from_pretrained(device=device)

# 3. Generate
print("Generating speech...")
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
