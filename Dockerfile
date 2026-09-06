# Movie Recap Bot — Docker image (Step A-F semantic engine)
#
# Packages the whole Python codebase. The container runs the recap CLI; point
# it at a movie via a mounted volume and (optionally) at an Ollama container
# for the LLM. Supabase pgvector credentials come from the environment.
#
# Build:      docker build -t movie-recap .
# Run (CLI):  docker run --rm -it \
#               -v "$PWD/movies:/movies:ro" -v "$PWD/output:/app/output" \
#               -e OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \
#               movie-recap auto --movie /movies/my_movie.mp4 --minutes 14 --name my-recap
# Run (UI):   see docker-compose.yml (Recap Studio + Ollama + optional Supabase env).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    OLLAMA_BASE_URL=http://localhost:11434/v1

# ffmpeg for audio rip + clipping + assembly (kept minimal, no X11 libs).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (better layer caching).
COPY movie-recap-bot/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt openai anthropic

# The whole bot package + configs + migrations.
COPY movie-recap-bot /app

# Where the pipeline writes its outputs (mount over this).
VOLUME ["/app/output", "/cache"]

ENTRYPOINT ["python", "-m", "recap.cli"]
CMD ["--help"]
