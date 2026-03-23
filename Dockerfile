FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg libgl1 git && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

RUN git clone --depth 1 https://github.com/antgroup/ditto-talkinghead.git /app/ditto-talkinghead

WORKDIR /app/ditto-talkinghead

RUN pip install \
    librosa==0.10.2.post1 \
    opencv-python-headless==4.10.0.84 \
    Pillow==11.0.0 \
    scikit-image==0.25.0 \
    imageio==2.36.1 \
    imageio-ffmpeg==0.5.1 \
    scipy==1.15.0 \
    numba==0.60.0 \
    soundfile==0.13.0 \
    filetype==1.2.0 \
    einops \
    tqdm==4.67.1 \
    Cython==3.0.11 \
    onnxruntime-gpu \
    requests \
    runpod \
    huggingface-hub \
    --no-cache-dir

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('digital-avatar/ditto-talkinghead', allow_patterns=['ditto_pytorch/*', 'ditto_cfg/*'], local_dir='/app/checkpoints')"

RUN python -c "from core.utils.blend import blend_images_cy; print('ok')"

ENV HF_HUB_OFFLINE=1

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
