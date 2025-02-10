import torch
import os
from flask import Flask, request, send_file
import io
import soundfile as sf
import numpy as np
import psutil
import threading
import time
import logging
from kokoro import KPipeline

app = Flask(__name__)

# Initialize the TTS pipeline
pipeline = KPipeline(lang_code='a')

# Attempt to reduce memory usage by converting the model to half precision.
if hasattr(pipeline, 'model'):
    try:
        pipeline.model = pipeline.model.half()
        logging.info("Converted TTS model to half precision (FP16).")
    except Exception as e:
        logging.warning(f"Could not convert model to half precision: {e}")

@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    text = data.get('text', '')
    audio_chunks = []
    
    # Use no_grad for inference and, if on GPU, use autocast for FP16
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                generator = pipeline(text, voice='af_heart', speed=1.2)
                for chunk in generator:
                    (_, _, audio) = chunk
                    audio_chunks.append(audio)
        else:
            generator = pipeline(text, voice='af_heart', speed=1.2)
            for chunk in generator:
                (_, _, audio) = chunk
                audio_chunks.append(audio)
    
    audio_out = np.concatenate(audio_chunks)
    
    with io.BytesIO() as wav_buffer:
        sf.write(wav_buffer, audio_out, 24000, format='WAV')
        wav_buffer.seek(0)
        return send_file(
            wav_buffer,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='audio.wav'
        )

def memory_monitor(interval=5):
    """Log current process RAM usage in MB at regular intervals."""
    process = psutil.Process()
    while True:
        mem = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        logging.info(f"Current RAM Usage: {mem:.2f} MB")
        time.sleep(interval)

def start_memory_logger():
    monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
    monitor_thread.start()

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # To ensure the memory monitor thread runs in only one process,
    # either disable the reloader or check if this is the main process.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
         start_memory_logger()
         
    # Disable the reloader by setting use_reloader=False.
    app.run(host='0.0.0.0', port=5000, threaded=False, debug=True, use_reloader=False)
