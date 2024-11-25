# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the prediction script and model
COPY scripts/predict.py .
COPY models/model_C=1.0.bin .

# Expose the port
EXPOSE 9696

# Set entry point (modificado para usar solo predict.py)
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
