FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY . .
CMD ["bash"]
