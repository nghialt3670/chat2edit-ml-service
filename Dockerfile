FROM nvidia/cuda:12.2.0-base-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
apt-get install -y ffmpeg libsm6 libxext6 python3 python3-pip nvidia-container-toolkit && \
rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt --verbose 

RUN pip install --no-cache-dir segment-anything==1.0

ENV CUDA_HOME=/usr/local/cuda \
    HF_HOME=./checkpoints/huggingface_hub \ 
    PYTHONPATH=/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
