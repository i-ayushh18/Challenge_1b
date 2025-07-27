FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel


COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

RUN python -m spacy download en_core_web_sm
ENV NLTK_DATA=/usr/local/nltk_data
RUN mkdir -p /usr/local/nltk_data && \
    python -m nltk.downloader -d /usr/local/nltk_data punkt


COPY . .

RUN if [ -f /app/docker_entrypoint.sh ]; then \
        sed -i 's/\r$//' /app/docker_entrypoint.sh && \
        chmod +x /app/docker_entrypoint.sh; \
    fi

ENTRYPOINT ["/bin/bash", "-c", "if [ -f /app/docker_entrypoint.sh ]; then exec /app/docker_entrypoint.sh \"$@\"; else echo 'Error: docker_entrypoint.sh not found!'; exit 1; fi", "--"]