# Dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime  # Updated to PyTorch 2.4.0 with CUDA 12.1

# Set working directory
WORKDIR /opt/program

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_DATASETS_CACHE=/tmp/hf_datasets_cache

# Install dependencies
COPY requirements.txt /opt/program/requirements.txt
RUN pip install --upgrade pip  # Ensure pip is up-to-date
RUN pip install --no-cache-dir -r /opt/program/requirements.txt

# Copy training code
COPY train.py /opt/program/
COPY evaluate.py /opt/program/

# Set entrypoint
ENTRYPOINT ["python", "train.py"]