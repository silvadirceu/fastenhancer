import requests
import base64
import json

# Token de inferência
token = "9bprJuV2"

# Caminho do áudio
audio_path = "dataset/010_orig.wav"

# Endpoint de predição
url = "http://127.0.0.1:8080/predictions/meu_modelo"

# Cabeçalhos
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "audio/wav"
}

# Envia o áudio como corpo da requisição
with open(audio_path, "rb") as f:
    response = requests.post(url, headers=headers, data=f)

# Verifica a resposta
print("Status:", response.status_code)
try:
    data = response.json()
    print("Resposta:", json.dumps(data, indent=2))

    # Salva os arquivos OGG decodificados
    with open("results/enhanced.ogg", "wb") as f:
        f.write(base64.b64decode(data["enhanced_ogg_base64"]))

    with open("results/residue.ogg", "wb") as f:
        f.write(base64.b64decode(data["residue_ogg_base64"]))

except Exception as e:
    print("Erro ao processar resposta:", e)
