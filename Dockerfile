# base image
FROM python:3.12

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# update timezone to display correct time
RUN ln -sf /usr/share/zoneinfo/Asia/Almaty /etc/localtime
RUN echo "Asia/Almaty" | tee /etc/timezone

# install system dependencies
RUN apt update && apt upgrade -y && \
    apt install -y ffmpeg vim && \
    rm -rf /var/lib/apt/lists/*

# copy requirements.txt files
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# set working directory
WORKDIR /app

# copy project files to the working directory
COPY . .

# command to run the application
ENTRYPOINT ["streamlit", "run", "chatui/chatui.py"]

