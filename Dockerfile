FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git tesseract-ocr && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker_entrypoint.sh /app/
RUN chmod +x /app/docker_entrypoint.sh

RUN python -m spacy download en_core_web_sm
RUN echo "spaCy model size:" && du -sh /usr/local/lib/python3.10/site-packages/en_core_web_sm*

ENV NLTK_DATA=/usr/local/nltk_data
RUN mkdir -p /usr/local/nltk_data
RUN python -m nltk.downloader -d /usr/local/nltk_data punkt

COPY . .

ENTRYPOINT ["/app/docker_entrypoint.sh"]