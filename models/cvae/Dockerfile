FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app

COPY . /app

RUN pip install google-cloud-storage numpy pandas

ENTRYPOINT ["python", "task.py"]
