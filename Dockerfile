# bayflux
#
# Version       0.3

FROM intelpython/intelpython3_core:2021.4.0-0 as build

LABEL maintainer="tbackman@lbl.gov"

USER root

# This command keeps APT from losing the http connection in Docker
RUN echo "Acquire::http::Pipeline-Depth 0;\n" >> /etc/apt/apt.conf

# install Debian packages
RUN apt-get update --allow-releaseinfo-change\
    && apt-get install -y \
        autoconf \
        automake \
        gcc \
        g++ \
        make \
        gfortran \
        wget \
        curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# compile mpich
# this line copied from ubuntu-mpi by Shane Canon
# https://github.com/NERSC/base-images/blob/master/ubuntu-mpi/Dockerfile
RUN cd /usr/local/src/ && \
    wget --no-check-certificate https://www.mpich.org/static/downloads/4.0.2/mpich-4.0.2.tar.gz && \
    tar xf mpich-4.0.2.tar.gz && \
    rm mpich-4.0.2.tar.gz && \
    cd mpich-4.0.2 && \
    ./configure && \
    make -j && make install && \
    cd /usr/local/src && \
    rm -rf mpich-4.0.2

# copy over bayflux 
RUN mkdir /bayflux
COPY . /bayflux/

# install Python packages
# note we re-install our custom cobrapy version AFTER installing bayflux
# because bayflux will get the regular version, which lacks the sampler
RUN pip install -U pip \
   && pip install -r /bayflux/requirements.txt --ignore-installed ruamel-yaml \
   && pip install -e /bayflux --ignore-installed ruamel-yaml

# remove bayflux code (latest version will be mounted at runtime)
RUN rm -rf /bayflux

# flatten image
FROM scratch
COPY --from=build / /
