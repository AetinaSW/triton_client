# based on miniconda3
FROM continuumio/miniconda3

# install dependent packages
RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install git -y
RUN apt-get install ffmpeg x264 libx264-dev -y

# Check ffmpeg
RUN ffmpeg -formats | grep h264

# Prepare InnoTIS 
RUN git clone https://github.com/MaxChangInnodisk/innotis-client.git
WORKDIR "/innotis-client"

# Create the conda environment:
RUN conda env create -f docker/environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "innotis", "/bin/bash", "-c"]

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "innotis", "python", "app.py"]
