# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Copy all project files
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Install spaCy English model (fixes 404 issue)
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz

RUN pip install python-multipart

# Pre-download and cache model inside the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder='/tmp/hf_cache')"
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
EXPOSE 7860

# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
