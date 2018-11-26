# Definition of Submission container

# We start from tensorflow-gpu image
FROM duckietown/pytorch-v0.4.1-py27-cuda9.0

# DO NOT MODIFY: your submission won't run if you do
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
         gcc \
         libc-dev\
         git \
         bzip2 \
         python-tk \
         libglib2.0-0 \
         libsm6 \
         libxrender1 \
         libfontconfig1 && \
     rm -rf /var/lib/apt/lists/*

# let's create our workspace, we don't want to clutter the container
RUN rm -r /workspace; mkdir /workspace

# set up conda
RUN conda create -y -n env python=3.6
RUN echo "source activate- env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

RUN conda install -n env -y -c pytorch pytorch=0.4.1

# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
COPY requirements.txt /workspace
RUN pip install -r /workspace/requirements.txt

# let's copy all our soluion files to our workspace
COPY features /workspace/features
COPY final_models /workspace/final_models
COPY models /workspace/models
COPY scripts /workspace/scripts
COPY training /workspace/training
COPY utils /workspace/utils

# we make the workspace our working directory
WORKDIR /workspace

# DO NOT MODIFY: your submission won't run if you do
ENV DUCKIETOWN_SERVER=evaluator

# let's see what you've got there...
CMD python -m scripts.solution
