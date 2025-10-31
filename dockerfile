# Base com OpenVINO e Python 3.11.13
FROM altran1502/immich-machine-learning:release-openvino

# Diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . /app

# Cria ambiente virtual com Python 3.11
RUN python3.11 -m venv /opt/venv

# Garante que o ambiente virtual seja usado por padrão
ENV PATH="/opt/venv/bin:$PATH"

# Atualiza pip e instala dependências
RUN pip install --upgrade pip

# Instala compilador e dependências de áudio
RUN apt-get update && apt-get install -y gcc python3.11-dev libsndfile1

# Instala torch e torchaudio compatíveis com CPU
RUN pip install torch==2.1.2+cpu torchaudio==2.1.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Instala demais dependências
RUN pip install --no-cache-dir -r requirements_freeze.txt

# Exponha a porta da API
EXPOSE 8000

# Comando para iniciar FastAPI
CMD ["uvicorn", "fastenhancer_api:app", "--host", "0.0.0.0", "--port", "8000"]
