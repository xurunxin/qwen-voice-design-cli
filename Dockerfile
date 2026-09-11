FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends sox libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir '.[server,model]'
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 QVD_HOME=/data QVD_MODEL_DIR=/models
EXPOSE 8096
ENTRYPOINT ["qvd"]
CMD ["serve", "--host", "0.0.0.0"]
