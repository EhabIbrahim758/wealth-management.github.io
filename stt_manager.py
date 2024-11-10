import pathlib
import numpy as np
import torch
from pydub import AudioSegment
from io import BytesIO



AUDIO_SAMPLE_RATE = 16000.0

REPO_ID = "facebook/seamless-m4t-v2-large"

MAX_INPUT_AUDIO_LENGTH = 500

CHECKPOINTS_PATH = pathlib.Path("/content/models")

    

def preprocess_audio(input_audio, sampling_rate=16000) -> None:
    audio = AudioSegment.from_file(BytesIO(input_audio))
    resampled_audio = audio.set_frame_rate(sampling_rate)
    samples = np.array(resampled_audio.get_array_of_samples())

    if resampled_audio.channels == 2:
        samples = samples.reshape((-1, 2))

    samples = samples.astype(np.float32) / 32768

    tensor = torch.tensor(samples)

    if resampled_audio.channels == 2:
        tensor = tensor.t()
    return tensor
    
def run_asr(translator, input_audio, target_language: str) -> str:
    target_language_code = target_language

    out_texts, _ = translator.predict(
        input=input_audio,
        task_str="ASR",
        src_lang=target_language_code,
        tgt_lang=target_language_code,
    )
    # print(out_texts[0])
    return str(out_texts[0])
