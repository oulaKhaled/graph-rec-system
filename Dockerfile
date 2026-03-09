# Use a Python base image
FROM python:3.11-slim


RUN apt-get update && apt-get install -y gcc g++ git libgomp1
# Set working directory
WORKDIR /app

# Copy and install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install torch matching your training environment
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# PyG packages matching torch 2.4.0 + cu118
RUN pip install torch_geometric
RUN pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

RUN pip install torchmetrics

# Copy the rest of the project
COPY . .

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]