FROM ubuntu:22.04

RUN apt-get -q update \
    && apt-get install -y \
    python3 python3-dev python3-pip \
    gcc gfortran binutils \
    hwloc libevent-dev \
    libudev-dev libnuma-dev libpciaccess-dev \
    openssh-client \
    git \
    && pip3 install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV MPI_DIR=/opt/ompi

ADD https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.bz2 .
RUN tar xf openmpi-4.1.6.tar.bz2 \
    && cd openmpi-4.1.6 \
    && ./configure --prefix=$MPI_DIR \
    && make -j4 all \
    && make install \
    && cd .. && rm -rf \
    openmpi-4.1.6 openmpi-4.1.6.tar.bz2 /tmp/*

# Add user and switch
RUN useradd -ms /bin/bash mpitest
USER mpitest
WORKDIR /home/mpitest

# Copy code into user's home
COPY --chown=mpitest:mpitest ./ /home/mpitest/pmmoto

ENV PATH="$MPI_DIR/bin:/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="$MPI_DIR/lib:"

CMD [ "tail", "-f", "/dev/null" ]