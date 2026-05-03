FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY examples ./examples

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .

ENTRYPOINT ["slam-eval"]
CMD ["--help"]

