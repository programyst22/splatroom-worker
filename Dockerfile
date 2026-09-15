FROM ghcr.io/nerfstudio-project/nerfstudio:latest

USER root
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY preflight.py /app/preflight.py

ENV PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SPLATROOM_WORKDIR=/tmp/splatroom

# Fail the image build early if the expected CLIs are missing.
RUN python /app/preflight.py --build-check

CMD ["python", "-u", "/app/handler.py"]
