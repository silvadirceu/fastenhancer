from pathlib import Path
import argparse
import os

import torch
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np

from utils import get_hparams, HParams
from wrappers import get_wrapper
from fastenhancer_class import TorchFastEnhancer, OpenVINOEnhancer, ONNXRunTimeEnhancer

def main():
    audio_path = "/home/carlosd/workspace/fastenhancer/dataset/010_orig.wav"
    model_path = "/home/carlosd/workspace/fastenhancer/logs/fastenhancer_l/"  # torch pth model
    #model_path = "/home/carlosd/workspace/fastenhancer/onnx/fastenhancer.default.spec.onnx"
    #model_path = "/home/carlosd/workspace/fastenhancer/onnx_models/fastenhancer_l.onnx"
    output_dir = "results/"
    os.makedirs(output_dir, exist_ok=True)

    fastEn = TorchFastEnhancer(model_dir=model_path, device="cpu")
    #fastEn = OpenVINOEnhancer(model_path=model_path, device="CPU")
    #fastEn = ONNXRunTimeEnhancer(model_path=model_path)
    
    # Load a noisy audio
    audio, fs = librosa.load(audio_path, sr=16000, mono=True)
    audio = torch.from_numpy(audio).float().to("cpu").unsqueeze(0)
    #audio = np.expand_dims(audio.astype(np.float32), axis=0)


    # Inference
    enhanced = fastEn.predict(x=audio)

    # Save the enhanced audio
    enhanced = enhanced.squeeze().cpu().numpy()
    audio_name = os.path.basename(audio_path).split(".")[0]
    outFileEnhanced = f"{output_dir}/{audio_name}_enhanced.wav"
    outFileResidue = f"{output_dir}/{audio_name}_residue.wav"
    sf.write(outFileEnhanced, enhanced, fs)

    residual = audio.squeeze().cpu().numpy() - enhanced
    sf.write(outFileResidue, residual, fs)

main()


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
