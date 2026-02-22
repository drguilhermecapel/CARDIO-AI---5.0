FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

# Copy necessary modules
COPY models /app/models
COPY utils /app/utils
COPY serving /app/serving

# Install dependencies
# scipy for signal processing, matplotlib for XAI viz utils, httpx for webhooks
RUN pip install fastapi uvicorn pydantic numpy scipy matplotlib httpx

# Expose port
EXPOSE 8080

# Run from serving directory
WORKDIR /app/serving
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
