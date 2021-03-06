# Definition of Submission container

# We start from tensorflow-gpu image
FROM duckietown/pytorch-v0.4.1-py27-cuda9.0

# DO NOT MODIFY: your submission won't run if you do
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         gcc \
         libc-dev\
         git \
         bzip2 \
         python-tk && \
     rm -rf /var/lib/apt/lists/*

# let's create our workspace, we don't want to clutter the container
RUN rm -r /workspace; mkdir /workspace

# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
COPY requirements.txt /workspace
RUN pip install -r /workspace/requirements.txt

# let's copy all our solution files to our workspace
COPY duckietown_rl /workspace/duckietown_rl
COPY final_models /workspace/final_models

# we make the workspace our working directory
WORKDIR /workspace

# DO NOT MODIFY: your submission won't run if you do
ENV DUCKIETOWN_SERVER=evaluator

# let's see what you've got there...
CMD python -m duckietown_rl.solution
