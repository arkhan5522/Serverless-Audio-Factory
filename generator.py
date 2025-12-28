# generator.py
import os
import subprocess
import sys

# ==========================================
# 1. AUTO-INSTALLER
# ==========================================
print("Installing dependencies...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chatterbox-tts", "--quiet"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchaudio", "--quiet"])
    # Ensure ffmpeg-python is installed if you want to use python wrapper, 
    # but here we will use subprocess to call system ffmpeg directly.
except Exception as e:
    print(f"Install Warning: {e}")

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

# ==========================================
# 3. DOWNLOAD REFERENCE AUDIO
# ==========================================
ref_audio_path = "reference_voice.mp3"
print(f"Downloading reference audio from: {AUDIO_URL}")

try:
    token = os.environ.get('GH_PAT', '')
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    headers['Accept'] = 'application/vnd.github.v3.raw'
    
    r = requests.get(AUDIO_URL, headers=headers, allow_redirects=True)
    
    if r.status_code == 200:
        with open(ref_audio_path, "wb") as f:
            f.write(r.content)
        print("Download successful.")
    else:
        print(f"Download failed: {r.status_code}")
        print(r.text[:200])
        exit(1)
        
except Exception as e:
    print(f"Error during download: {e}")
    exit(1)

# ==========================================
# 4. LOAD MODEL
# ==========================================
print("Loading Chatterbox model...")
try:
    model = ChatterboxTTS.from_pretrained(device=device)
except Exception as e:
    print(f"Model Load Error: {e}")
    exit(1)

# ==========================================
# 5. TEXT SEGMENTATION & GENERATION
# ==========================================
def chunk_text(text, limit=250):
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= limit:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

print("Segmenting text...")
text_segments = chunk_text(TEXT_TO_SPEAK, limit=250)
print(f"Total segments to process: {len(text_segments)}")

generated_files = []

try:
    with torch.no_grad():
        for i, segment in enumerate(text_segments):
            print(f"Processing segment {i+1}/{len(text_segments)}...")
            
            # Generate audio for this chunk
            wav = model.generate(
                text=segment,
                audio_prompt_path=ref_audio_path,
                exaggeration=0.5,
                cfg_weight=0.5
            )
            
            # Save individual chunk
            chunk_filename = f"chunk_{i}.wav"
            ta.save(chunk_filename, wav.cpu(), model.sr)
            generated_files.append(chunk_filename)
            print(f"Saved {chunk_filename}")

except Exception as e:
    print(f"Generation Error: {e}")
    # Clean up creates files if it fails mid-way
    for f in generated_files:
        if os.path.exists(f):
            os.remove(f)
    exit(1)

# ==========================================
# 6. COMBINE AUDIO (FFMPEG)
# ==========================================
print("Combining segments with FFmpeg...")

list_file = "file_list.txt"
output_filename = "output.wav"

try:
    # 1. Create a text file listing all chunks for FFmpeg
    with open(list_file, "w") as f:
        for filename in generated_files:
            f.write(f"file '{filename}'\n")

    # 2. Run FFmpeg concat command
    # -f concat: use the concat demuxer
    # -safe 0: allow unsafe file paths (standard practice for local scripts)
    # -c copy: copy stream without re-encoding (very fast)
    cmd = [
        "ffmpeg", 
        "-f", "concat", 
        "-safe", "0", 
        "-i", list_file, 
        "-c", "copy", 
        "-y",  # Overwrite output if exists
        output_filename
    ]
    
    subprocess.check_call(cmd)
    print(f"SUCCESS: Full audio saved to {output_filename}")

except subprocess.CalledProcessError as e:
    print(f"FFmpeg Merge Error: {e}")
except Exception as e:
    print(f"General Merge Error: {e}")

finally:
    # Cleanup temporary chunks and list file
    print("Cleaning up temporary files...")
    if os.path.exists(list_file):
        os.remove(list_file)
    for f in generated_files:
        if os.path.exists(f):
            os.remove(f)
