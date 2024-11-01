FROM python:3.12


# Set the working directory to /app
WORKDIR /main


# Copy the current directory contents into the container at /main
COPY . /main


# Install the basic packages to run opencv
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev


# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt



# Run main.py whe the container launches
# ENTRYPOINT python /main/main.py 101_1.tif 102_1.tif
ENTRYPOINT ["/main/run.sh"]




LABEL authors="rafas"

ENTRYPOINT ["top", "-b"]