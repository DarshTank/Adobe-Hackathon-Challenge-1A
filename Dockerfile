FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Process.py .

RUN mkdir -p /app/input /app/output

ENV PDF_INPUT_DIR=/app/input
ENV PDF_OUTPUT_DIR=/app/output

CMD ["python", "Process.py"]
