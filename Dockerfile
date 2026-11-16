FROM python:3.12-slim
WORKDIR /app

# WeasyPrint needs pango, harfbuzz, and the Devanagari font stack for the Hindi PDF.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libharfbuzz0b \
        libffi8 \
    && rm -rf /var/lib/apt/lists/*

COPY intake/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY intake /app/intake
COPY .git /app/.git
COPY packs /app/packs
COPY scripts /app/scripts

ENV PYTHONPATH=/app
ENV DATA_DIR=/data
EXPOSE 8000

CMD ["sh", "-c", "uvicorn intake.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
