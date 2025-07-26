FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install system dependencies with retry logic
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Download and install language models during build
RUN python -m spacy download en_core_web_sm
ENV NLTK_DATA=/usr/local/nltk_data
RUN mkdir -p /usr/local/nltk_data && \
    python -m nltk.downloader -d /usr/local/nltk_data punkt

# Copy the application code
COPY . .

# Ensure the entrypoint script has Unix line endings and is executable
RUN if [ -f /app/docker_entrypoint.sh ]; then \
        sed -i 's/\r$//' /app/docker_entrypoint.sh && \
        chmod +x /app/docker_entrypoint.sh; \
    fi

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "-c", "if [ -f /app/docker_entrypoint.sh ]; then exec /app/docker_entrypoint.sh \"$@\"; else echo 'Error: docker_entrypoint.sh not found!'; exit 1; fi", "--"]