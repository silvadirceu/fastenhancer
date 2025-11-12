import torch

from utils import get_hparams
from wrappers import get_wrapper

# 1️⃣ Carrega a configuração e o wrapper
model_dir = "/home/carlosd/workspace/fastenhancer/logs/fastenhancer_l/"  # torch pth model

device = torch.device('cpu')
hps = get_hparams(base_dir=model_dir)
wrapper = get_wrapper(hps.wrapper)(hps, device=device)
model = wrapper.model
model.eval()

# 3️⃣ Prepara uma entrada dummy (1 batch, 1 canal, ~1s de áudio a 16kHz)
dummy_input = torch.randn(1, 1, 16000)

# 4️⃣ Cria o modelo TorchScript
traced = torch.jit.trace(model, dummy_input)

# 5️⃣ Salva em formato .pt
traced.save("fastenhancer.pt")

print("✅ Modelo salvo em fastenhancer.pt")
