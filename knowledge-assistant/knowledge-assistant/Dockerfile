FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system build dependencies required by some packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       g++ \
       cmake \
       poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

EXPOSE 8501

# Run ingestion then start Streamlit (listen on all interfaces)
CMD ["/bin/bash", "-lc", "python src/ingest.py && streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0"]
