# Medical Image Analysis System - Docker Configuration

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and medical imaging
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    python3-tk \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p uploads outputs static/css static/js templates models utils

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5000/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "--access-logfile", "-", "app:app"]
