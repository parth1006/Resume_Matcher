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

# ✅ Create writable directory in /tmp (HF Spaces allows writes here)
RUN mkdir -p /tmp/data/resumes && chmod -R 777 /tmp/data

EXPOSE 7860

# ✅ Remove BASE_DIR env variable - let the app handle it based on SPACE_ID
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]