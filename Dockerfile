FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 🔥 Add this line to download NLTK data inside container
RUN python -m nltk.downloader stopwords wordnet

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]

