# FROM aliyun-va-hub.byted.org/base/lab.pytorch2:2.5.1.py311.cu124.post1
FROM aliyun-va-hub.byted.org/base/kg_uprof_b200_test:a579b66f8e26029d41d023b041fbf08e
# py3.11 torch2.7.1+cu12.8

# Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive
# ENV GRADIO_SERVER_NAME=0.0.0.0
# ENV GRADIO_SERVER_PORT=7860

# Install system dependencies
# ENV TZ=Asia/Shanghai PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ PIP_TRUSTED_HOST=mirrors.aliyun.com

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    wget \
    vim \
    curl libgl1 \
    libglib2.0-0 \
    openssh-server \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up SSH server
RUN mkdir -p /run/sshd && \
    chmod 755 /run/sshd && \
    /usr/sbin/sshd

# Set working directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Copy requirements first for better caching
# COPY requirements.txt .

RUN git clone https://github.com/ZachL1/Bagel.git

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel nvitop
RUN sed -i '/torch/d' Bagel/requirements.txt
RUN pip install -r Bagel/requirements.txt

# RUN pip install torch==2.7.1 torchvision torchaudio==2.7.1  --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
# RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.7.4.post1+cu126torch2.7-cp311-cp311-linux_x86_64.whl
# RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.18/flash_attn-2.7.4+cu124torch2.5-cp311-cp311-linux_x86_64.whl
# RUN pip uninstall -y opencv-python numpy scipy && pip install opencv-python-headless numpy scipy
RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu128torch2.7-cp311-cp311-linux_x86_64.whl

# Copy the entire project
# COPY . .


# # Create models directory
# RUN mkdir -p models

# # Expose port
# EXPOSE 7860

# Default command to run the Gradio app
# CMD ["python", "app.py", "--server_name", "0.0.0.0", "--server_port", "7860", "--mode", "2"]
# CMD /bin/bash