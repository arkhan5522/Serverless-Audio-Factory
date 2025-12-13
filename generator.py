# generator.py
import base64
import os
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# --- INJECTED_VARIABLES_START ---
# The GitHub Action will replace these dummy values
TEXT_TO_SPEAK = """{{TEXT_PLACEHOLDER}}"""
AUDIO_B64 = """{{AUDIO_B64_PLACEHOLDER}}"""
# --- INJECTED_VARIABLES_END ---

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Decode the Reference Audio
ref_audio_path = "reference_voice.mp3"
with open(ref_audio_path, "wb") as f:
    f.write(base64.b64decode(AUDIO_B64))

# 3. Load Model & Audio
model = ChatterboxTTS.from_pretrained(device=device)
reference_audio, sr = ta.load(ref_audio_path)

# 4. Generate Speech (Simple chunking if needed, but keeping it simple for speed)
# For very long text, you'd want the chunking function from your original script.
print("Generating speech...")
with torch.no_grad():
    wav = model.generate(
        text=TEXT_TO_SPEAK,
        audio_prompt_path=ref_audio_path,
        exaggeration=0.5,
        cfg_weight=0.5
    )

# 5. Save Output
output_filename = "output.wav"
ta.save(output_filename, wav.cpu(), model.sr)
print("Done.")
