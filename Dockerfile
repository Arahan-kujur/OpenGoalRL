# Reproducible OpenGoalRL environment with all GRF system dependencies baked in.
# This removes the #1 onboarding failure: building gfootball's C++/SDL stack.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# GRF build/runtime system dependencies (mirrors CONTRIBUTING.md / README).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
        libsdl2-dev \
        libsdl2-image-dev \
        libsdl2-ttf-dev \
        libsdl2-gfx-dev \
        libboost-all-dev \
        libfontconfig1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml setup.py README.md ./
COPY opengoalrl ./opengoalrl

# Full install (includes gfootball); this is the heavy step.
RUN pip install --upgrade pip && pip install -e .

# Copy the rest of the project (benchmarks, configs, docs, notebooks).
COPY . .

CMD ["python", "-m", "opengoalrl.scripts.train", "--config", "opengoalrl/configs/empty_goal_close.yaml"]
