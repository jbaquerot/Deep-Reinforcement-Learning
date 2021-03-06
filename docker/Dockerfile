FROM ubuntu:latest

MAINTAINER Jose Carlos Baquero <jbaquerot@gmail.com>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        #python \
        #python-dev \
        rsync \
        software-properties-common \
        unzip \
	git \
	cmake \
	zlib1g-dev \
	libjpeg-dev \
	xvfb \
	libav-tools \
	xorg-dev \
	python-opengl \
	libboost-all-dev \
	libsdl2-dev \
	swig \
	python3 \
	python3-dev \
	python3-setuptools \
	python3-pip \
    python3-tk \
	ipython3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
#    python get-pip.py && \
#    rm get-pip.py

RUN pip3 install --upgrade pip

RUN pip3 --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
	future \
	theano 
	#\
    #    && \
    #python -m ipykernel.kernelspec


# jupyter for python 3
#RUN python3 -m pip install ipykernel
#RUN python3 -m ipykernel install --user

# Install TensorFlow CPU version from central repo
RUN pip install tensorflow 

# Install OpenAI Gym
RUN pip install 'gym[all]'

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /root

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

RUN mkdir "/notebooks"
WORKDIR "/notebooks"

RUN ["chmod", "+x", "/root/run_jupyter.sh"]
CMD ["/root/run_jupyter.sh", "--allow-root"]
