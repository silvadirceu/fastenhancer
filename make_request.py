import requests
import base64

with open("dataset/010_orig.wav", "rb") as f:
    response = requests.post("http://localhost:8000/predict", files={"file": f})

data = response.json()

# Salva os arquivos OGG localmente
with open("enhanced.ogg", "wb") as f:
    f.write(base64.b64decode(data["enhanced_ogg_base64"]))

with open("residue.ogg", "wb") as f:
    f.write(base64.b64decode(data["residue_ogg_base64"]))
