import torch
import numpy as np
import pickle
from ts.torch_handler.base_handler import BaseHandler


class FastEnhancerHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def preprocess(self, data):
        """Desserializa os dados de entrada."""
        # Espera bytes pickleados (ex: enviados pelo cliente gRPC)
        if isinstance(data, list) and "data" in data[0]:
            audio = pickle.loads(data[0]["data"])
        else:
            audio = data

        if isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)

        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)  # (1, N)
        tensor = torch.from_numpy(audio).unsqueeze(1).to(self.device)  # (B, 1, N)

        return tensor

    def inference(self, tensor):
        """Executa a inferência usando o modelo TorchScript."""
        with torch.no_grad():
            output = self.model(tensor)
            if isinstance(output, tuple):
                enhanced = output[0]
            else:
                enhanced = output

        return enhanced.cpu().numpy().squeeze()

    def postprocess(self, result):
        """Serializa o resultado para envio via gRPC/HTTP."""
        data = {"enhanced": result.tolist()}
        return [pickle.dumps(data)]
