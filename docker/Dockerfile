# app/Dockerfile

# Specify the base image
FROM python:3.10


# Set environment variables for UTF-8 support
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Update and install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# install pip and torch   
RUN pip install --upgrade pip
#RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to cache dependencies layer
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install -r requirements.txt

# copy all elements
COPY . /app
 
# listen to port 8501
EXPOSE 8501

# Set the environment variable for Streamlit static files
ENV STREAMLIT_STATIC_PATH=/dataset/pdf

# Healthcheck command to ensure the container is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# run the streamlit run command
ENTRYPOINT ["streamlit", "run", "./app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]