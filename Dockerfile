# app/Dockerfile

# Specify the base image
FROM ubuntu:18.04
FROM python:3.10.11
 

# Set environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

 

# Define the workdir path
WORKDIR ./app

 

# upgrade all the packages that have updates available and
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*



# install pip
RUN pip3 install --upgrade pip

 

# copy all elements
COPY . .

 
# install all packages in requierements.txt
RUN pip3 install -r requirements.txt

 
# listen to port 8501
EXPOSE 8501

 

# check if the container still working
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

 

# run the streamlit run command
ENTRYPOINT ["streamlit", "run", "./app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


