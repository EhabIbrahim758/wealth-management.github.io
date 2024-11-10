import asyncio
 
import websockets
 
import torch
import numpy as np
from translator import Translator
from stt_manager import run_asr, preprocess_audio

from pydub import AudioSegment
from io import BytesIO



if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

translator = Translator(
    model_name_or_card="seamlessM4T_v2_large",  # Specifies the model to be used.
    vocoder_name_or_card="vocoder_v2",          # Specifies the vocoder to be used.
    device=device,                              # Sets the device (CPU or CUDA-enabled GPU).
    dtype=dtype,                                # Sets the data type (float16 for GPU, float32 for CPU).
    apply_mintox=True,                          # Applies minimum toxicity constraints if available.
)

  
async def handler(websocket, path):
    audio_blob = await websocket.recv()
    audio_blob = preprocess_audio(audio_blob)
    text = run_asr(translator, audio_blob,  "eng")
    await websocket.send(text)
 
 
if __name__ == "__main__":   
    start_server = websockets.serve(handler, "localhost", 8000)
    # torch.cuda.empty_cache()
    asyncio.get_event_loop().run_until_complete(start_server)
    # torch.cuda.empty_cache()
    asyncio.get_event_loop().run_forever()
    # torch.cuda.empty_cache()