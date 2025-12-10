# Use Python 3.12 slim
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libx11-6 \
    libxrender1 \
    libxext6 \
    libxtst6 \
    libxi6 \
    tk \
    tcl \
    python3-pyqt5 \
    qtbase5-dev \
    libqt5widgets5 \
    libqt5gui5 \
    libqt5core5a \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/psecola/ECE-792-CT-Image.git

# Set working directory to repo root
WORKDIR /app/ECE-792-CT-Image

# Remove invalid collections line and install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory for running the script
WORKDIR /app/ECE-792-CT-Image

# Add PYTHONPATH to include modules
ENV PYTHONPATH=/app/ECE-792-CT-Image

# Run PyVista script
CMD ["python", "CT_annotation_tool/CT_PyVista_PC.py"]
