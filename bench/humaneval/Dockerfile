# Isolated runner for model-generated Python. No network at run time (the sandbox passes --network none);
# packages are installed here, at build time, and pinned so results are reproducible.
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.3.5 pytest-json-report==1.5.0 \
    && useradd --create-home --uid 10001 runner
USER runner
WORKDIR /home/runner
