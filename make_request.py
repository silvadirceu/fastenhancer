import requests
import base64
import os

with open("dataset/010_orig.wav", "rb") as f:
    response = requests.post("http://localhost:8000/predict", files={"file": f})

data = response.json()

# Salva os arquivos OGG localmente
os.makedirs("results/", exist_ok=False)
with open("enhanced.ogg", "wb") as f:
    f.write(base64.b64decode(data["results/enhanced_ogg_base64"]))

with open("residue.ogg", "wb") as f:
    f.write(base64.b64decode(data["results/residue_ogg_base64"]))
