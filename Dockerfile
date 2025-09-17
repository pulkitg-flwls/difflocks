FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    ENV_NAME=difflocks

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git cmake wget curl unzip \
    build-essential libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
libegl1 \
libgl1-mesa-glx \
libx11-6 \
libgl1 \
libglvnd0 \
libglx0 \
libopengl0 \
&& rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    p7zip-full \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda first, then install mamba
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

# Accept conda Terms of Service and install mamba
RUN $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    $CONDA_DIR/bin/conda install -n base -c conda-forge mamba -y

ENV PATH=$CONDA_DIR/bin:$PATH

# Make mamba available in non-interactive shells
SHELL ["/bin/bash", "-c"]
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "mamba activate $ENV_NAME" >> ~/.bashrc

# Copy and create mamba environment
COPY environment.yml /tmp/environment.yml
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    mamba env create -f /tmp/environment.yml -c conda-forge && \
    mamba shell init --shell bash --root-prefix=$CONDA_DIR && \
    echo "mamba activate $ENV_NAME" >> ~/.bashrc

# Install extra pip packages inside the env
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    eval "$(mamba shell hook --shell bash)" && \
    mamba activate $ENV_NAME && \
    pip install ninja && \
    pip install flash-attn==2.4.2 --no-build-isolation

# Copy the entire project directory into the container
COPY . /app/projects/difflocks/

RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    eval "$(mamba shell hook --shell bash)" && \
    mamba activate $ENV_NAME && \
    mkdir -p /app/projects/difflocks/externals && \
    git clone --recursive https://github.com/SHI-Labs/NATTEN.git /app/projects/difflocks/externals/natten && \
    cd /app/projects/difflocks/externals/natten && \
    TORCH_CUDA_ARCH_LIST="9.0" NATTEN_CUDA_ARCH="9.0" FORCE_CUDA=1 python setup.py install
    # pip install git+https://github.com/SHI-Labs/NATTEN.git

RUN apt-get update && apt-get install -y p7zip-full
RUN curl -L -o /usr/local/bin/7zz https://www.7-zip.org/a/7zz && \
    chmod +x /usr/local/bin/7zz

ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    chown -R $USERNAME:$USERNAME /app /opt/conda

USER $USERNAME
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> /home/$USERNAME/.bashrc && \
    echo "mamba activate $ENV_NAME" >> /home/$USERNAME/.bashrc

# Set workdir and default PYTHONPATH
WORKDIR /app/projects/difflocks
ENV HOME=/home/$USERNAME
ENV PYTHONPATH=/app/projects

# Default shell will drop into the activated conda environment
ENTRYPOINT ["/bin/bash", "--login"]