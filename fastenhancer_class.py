from pathlib import Path
import argparse
import os

import torch
import librosa
import soundfile as sf
from tqdm import tqdm

from utils import get_hparams, HParams
from wrappers import get_wrapper


import onnxruntime as ort
import numpy as np

class ONNXRunTimeEnhancer:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)
        result = self.session.run([self.output_name], {self.input_name: x})[0]
        return result.squeeze()


from openvino import Core
import numpy as np

class OpenVINOEnhancer:
    def __init__(self, model_path: str, device: str = "CPU"):
        core = Core()
        self.model = core.read_model(model_path)
        self.compiled_model = core.compile_model(self.model, device)
        self.input_layer = self.model.inputs[0]
        self.output_layer = self.model.outputs[0]
        self.device = device

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        result = self.compiled_model([x])[self.output_layer]
        return result.squeeze()



class TorchFastEnhancer:

    model_dir:str
    device:str

    def __init__(self, model_dir, device="cpu"):
        self.model_dir = model_dir
        #device = torch.device("cuda")
        self.device = torch.device(device.lower())
        self.hps = self.get_params(self.model_dir)
        self.model = self.load_model(self.hps, self.device)

    def get_params(self, model_dir:str) -> HParams:
        hps = get_hparams(base_dir=model_dir)
        print(f"HPS.wrapper: {hps.wrapper}")
        return hps

    def load_model(self, hps:HParams, device):
        wrapper = get_wrapper(hps.wrapper)(hps, device=device)
        wrapper.load()
        return wrapper
    
    def predict(self, x:torch.Tensor):
        with torch.no_grad():
            enhanced, spec = self.model.model(x)  # return: wav, spec
        return enhanced
        
    


# def main():
#     audio_path = "/home/carlosd/workspace/fastenhancer/dataset/010_orig.wav"
#     model_dir = "/home/carlosd/workspace/fastenhancer/logs/fastenhancer_l/"
#     output_dir = "results/"
#     os.makedirs(output_dir, exist_ok=True)

#     fastEn = FastEnhancer(model_dir=model_dir, device="cpu")

#     # Load a noisy audio
#     audio, fs = librosa.load(audio_path, sr=16000, mono=True)
#     audio = torch.from_numpy(audio).float().to(fastEn.device).unsqueeze(0)

#     # Inference
#     enhanced = fastEn.predict(x=audio)

#     # Save the enhanced audio
#     enhanced = enhanced.squeeze().cpu().numpy()
#     audio_name = os.path.basename(audio_path).split(".")[0]
#     outFileEnhanced = f"{output_dir}/{audio_name}_enhanced.wav"
#     outFileResidue = f"{output_dir}/{audio_name}_residue.wav"
#     sf.write(outFileEnhanced, enhanced, fs)

#     residual = audio.squeeze().cpu().numpy() - enhanced
#     sf.write(outFileResidue, residual, fs)

# main()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('test model')
#     parser.add_argument(
#         '-n', '--name',
#         type=str,
#         required=True,
#         help='The latest checkpoint in logs/{name} will be loaded.'
#     )
#     parser.add_argument(
#         '-i', '--input-dir',
#         type=str,
#         default='/home/shahn/Datasets/DNS-Challenge/16khz/testset_synthetic_interspeech2020/no_reverb/noisy',
#         help='The dir path including noisy wavs for evaluation.'
#     )
#     parser.add_argument(
#         '-o', '--output-dir',
#         type=str,
#         default='enhanced/dns',
#         help='The dir path to save enhanced wavs.'
#     )

#     args = parser.parse_args()
#     main(args)
