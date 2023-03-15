FROM python:3.10.10

# Install production dependencies.
ADD requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

# Copy local code to the container image.
WORKDIR /app
COPY . .

ENV PORT 8000
ENV NAME auditmoduleapi
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 0 app:app