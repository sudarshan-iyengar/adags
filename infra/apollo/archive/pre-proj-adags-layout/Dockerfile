# ======================================================================
# Base image (Determined AI, CUDA 11.8, PyTorch 2.0)
# ======================================================================
FROM determinedai/environments:cuda-11.8-pytorch-2.0-gpu-0.31.1

# ======================================================================
# Set working directory and environment variables
# ======================================================================
WORKDIR /tmp
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=/usr/local/cuda-11.8/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9"
ENV MAX_JOBS=4

# ======================================================================
# Install system packages
# ======================================================================
RUN apt-get update && apt-get install -y \
    git \
    wget \
    colmap \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ======================================================================
# Install Python dependencies
# ======================================================================
RUN pip install --no-cache-dir \
    plyfile==0.8.1 \
    tqdm==4.66.1 \
    mediapy==1.1.2 \
    torchmetrics==0.11.4 \
    imagesize==1.4.1 \
    kornia==0.6.12 \
    omegaconf==2.3.0 \
    lpips

# ======================================================================
# Copy and build local CUDA extensions inside image
# ======================================================================
COPY simple-knn /tmp/simple-knn
COPY pointops2 /tmp/pointops2

# Install simple-knn
RUN cd /tmp/simple-knn && \
    pip install --no-cache-dir . && \
    python -c "import simple_knn; print('✓ simple_knn installed successfully')"

# Install pointops2
RUN cd /tmp/pointops2 && \
    pip install --no-cache-dir . && \
    python -c "import pointops; print('✓ pointops installed successfully')" || \
    python -c "import pointops2; print('✓ pointops2 installed successfully (module name: pointops2)')"

# ======================================================================
# Verify PyTorch and CUDA
# ======================================================================
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# List installed packages for debugging
RUN pip list | grep -E "(simple|point)"

# ======================================================================
# Default working directory
# ======================================================================
WORKDIR /apollo/sri/adags