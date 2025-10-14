    # Use official lightweight Python image
    FROM python:3.10-slim

    # Set working directory
    WORKDIR /code
    COPY . .

    # Install system dependencies
    RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

    # Install dependencies
    RUN pip uninstall -y fitz || true
    RUN pip install --no-cache-dir -r requirements.txt

    # ✅ Install spaCy English model (fixes 404 issue)
    RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz

    RUN python -m nltk.downloader stopwords

    # Pre-download and cache model inside the image
    RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder='/tmp/hf_cache')"
    ENV TRANSFORMERS_CACHE=/tmp/hf_cache
    EXPOSE 7860
    RUN mkdir -p /tmp/data/resumes
    ENV BASE_DIR=/tmp/data
    CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
