# Use uma imagem base do Python com suporte a Tkinter
FROM python:3.9-slim

# Instala as dependências do sistema necessárias para OpenCV, Tkinter e GUI
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 python3-tk && \
    apt-get clean

# Define o diretório de trabalho
WORKDIR /app

# Copia o arquivo requirements.txt para o contêiner
COPY requirements.txt .

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante da aplicação para o contêiner
COPY . .

# Define a variável de ambiente DISPLAY (necessário para GUI)
ENV DISPLAY=:0

# Comando para executar o aplicativo
CMD ["python", "main.py"]
