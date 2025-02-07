FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV SHELL /bin/bash
# Hugging Face
ENV HF_HOME "/runpod-volume/.cache/huggingface/"
ENV HF_HUB_ENABLE_HF_TRANSFER "1"
# pip
ENV PIP_CACHE_DIR "/runpod-volume/.cache/pip"
ENV PIP_PREFER_BINARY 1
ENV PYTHONUNBUFFERED 1 
# uv
ENV PYTHON_VERSION 3.11
ENV HOME /root
ENV PATH "$HOME/.local/bin/:$PATH"

WORKDIR /

# Update, upgrade, install packages and clean up
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    # Basic
    apt install --yes --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    file \
    git \
    git-lfs \
    inotify-tools \
    jq \
    libgl1 \
    lsof \
    vim \
    nano \
    # SSH access
    openssh-server \
    procps \
    rsync \
    sudo \
    software-properties-common \
    unzip \
    wget \
    zip && \
    # Build Tools and Development
    apt install --yes --no-install-recommends \
    build-essential \
    make \
    cmake \
    gfortran \
    libblas-dev \
    libbz2-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    liblzma-dev \
    libsqlite3-dev \
    liblapack-dev && \
    # Deep Learning Dependencies and Miscellaneous
    apt install --yes --no-install-recommends \
    libatlas-base-dev \
    libffi-dev \
    libhdf5-serial-dev \
    libsm6 \
    nvtop \
    task-spooler \
    libssl-dev && \
    # File Systems and Storage
    apt install --yes --no-install-recommends \
    cifs-utils \
    nfs-common \
    zstd &&\
    # Cleanup
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Set locale
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# uv
RUN sh -c "$(curl -fsSL https://astral.sh/uv/install.sh)"

# gcloud CLI
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y


# zsh
RUN sh -c "$(curl -L https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)"

# Remove existing SSH host keys
RUN rm -f /etc/ssh/ssh_host_*

# Start Scripts
COPY scripts/start.sh /
COPY scripts/post_start.sh /
RUN chmod +x /start.sh && \
    chmod +x /post_start.sh

CMD [ "/start.sh" ]