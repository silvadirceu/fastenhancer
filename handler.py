import io
import base64
import torch
import librosa
import soundfile as sf
import numpy as np
from ts.torch_handler.base_handler import BaseHandler

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from wrappers import get_wrapper
from utils import get_hparams, HParams


class FastEnhancerHandler(BaseHandler):
    """
    TorchServe handler para o modelo FastEnhancer.
    Retorna áudios enhanced e residue em Base64 (OGG).
    """

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        """Carrega o modelo e parâmetros a partir do diretório do TorchServe."""
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        # Config e pesos
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hps = get_hparams(base_dir=model_dir)
        WrapperClass = get_wrapper(self.hps.wrapper)
        self.wrapper = WrapperClass(self.hps, device=self.device)
        self.wrapper.load()  # carrega o checkpoint .pth
        self.wrapper.model.eval()
        self.initialized = True
        print("✅ FastEnhancer inicializado com sucesso.")

    def preprocess(self, data):
        """Converte o arquivo de áudio recebido em tensor PyTorch."""
        audio_bytes = data[0].get("body")

        # Leitura de bytes WAV
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        x = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        return x, sr

    def inference(self, data, *args, **kwargs):
        """Executa inferência no modelo."""
        x, sr = data
        with torch.no_grad():
            enhanced, _ = self.wrapper.model(x)
        enhanced = enhanced.squeeze().cpu().numpy()
        residue = x.squeeze().cpu().numpy() - enhanced
        return enhanced, residue, sr

    def postprocess(self, inference_output):
        """Converte enhanced e residue em Base64 (OGG) e retorna JSON."""
        enhanced, residue, sr = inference_output

        def to_b64(audio):
            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="OGG")
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")

        return [{
            "enhanced_ogg_base64": to_b64(enhanced),
            "residue_ogg_base64": to_b64(residue),
            "sample_rate": sr
        }]
