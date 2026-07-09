# TINAT — reproducible container for the end-to-end fusion demo.
#
# The demo and the fusion/metric logic depend only on the Python standard
# library, so a slim base image is enough and builds in seconds. The heavy ML
# stack (torch, tensorflow, ultralytics) is intentionally NOT installed here:
# it is only needed for real training/inference, not for this demo. See
# docs/decision-log.md.

FROM python:3.10-slim

# Stream stdout in real time and skip .pyc files inside the container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy the project (see .dockerignore for what is excluded).
COPY . .

# Run the end-to-end demo by default: mock detector -> fusion -> JSON export.
CMD ["python", "-m", "src.demo"]
