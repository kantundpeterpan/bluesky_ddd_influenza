FROM python:3.13-slim-bookworm

# Install necessary tools
RUN apt-get update && \
    apt-get install -y git curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/

# Clone the repository
RUN git clone https://github.com/kantundpeterpan/bluesky_ddd_influenza.git /project

# Install Python dependencies
WORKDIR /project
RUN uv pip install -r requirements.txt --system

# Set the working directory
WORKDIR /project/pipelines/bsky_housekeeping

# Define the entry point (optional, if only running one command)
# ENTRYPOINT ["python", "bsky_housekeeping_pipeline.py"]