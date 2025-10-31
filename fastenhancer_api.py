from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import librosa
import soundfile as sf
import os
import io
import base64
from utils import get_hparams, HParams
from wrappers import get_wrapper
from fastenhancer_class import FastEnhancer

app = FastAPI()
model_dir = "logs/fastenhancer_l"
enhancer = FastEnhancer(model_dir=model_dir, device="cpu")

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    # Lê o conteúdo do arquivo como bytes
    audio_bytes = await file.read()
    audio_stream = io.BytesIO(audio_bytes)

    # Carrega com soundfile
    audio, fs = sf.read(audio_stream, dtype='float32')

    # Resample se necessário
    if fs != 16000:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=16000)
        fs = 16000

    # Prepara tensor
    audio_tensor = torch.from_numpy(audio).float().to(enhancer.device).unsqueeze(0)

    # Inference
    enhanced = enhancer.predict(x=audio_tensor).squeeze().cpu().numpy()
    residual = audio - enhanced

    # Converte os resultados para OGG em memória
    def to_base64_ogg(audio_array):
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, fs, format='OGG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    enhanced_b64 = to_base64_ogg(enhanced)
    residue_b64 = to_base64_ogg(residual)

    return JSONResponse(content={
        "enhanced_ogg_base64": enhanced_b64,
        "residue_ogg_base64": residue_b64
    })
