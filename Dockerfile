FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG DOCKER_NAME
ARG DOCKER_USER_ID
ARG DOCKER_GROUP_ID

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    git-lfs \
    pkg-config \
    python3-dev \
    python3.10-venv \
    sudo \
    tmux \
    unzip \
    vim \
    wkhtmltopdf \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1 \
    # pip:
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_ROOT_USER_ACTION=ignore

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN groupadd -g $DOCKER_GROUP_ID $DOCKER_NAME

RUN adduser --disabled-password --uid $DOCKER_USER_ID --gid $DOCKER_GROUP_ID --gecos '' --shell /bin/bash $DOCKER_NAME \
 && chown -R $DOCKER_NAME:$DOCKER_NAME /app

RUN echo "$DOCKER_NAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-$DOCKER_NAME

USER $DOCKER_NAME
WORKDIR /app
RUN echo "source /app/.venv/bin/activate" >> ~/.bashrc
