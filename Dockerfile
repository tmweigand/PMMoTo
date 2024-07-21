FROM ubuntu:22.04

RUN apt-get -q update \
    && apt-get install -y \
    python3 python3-dev python3-pip \
    gcc gfortran binutils \
    && pip3 install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV MPI_DIR=/opt/ompi

ADD https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.0.tar.bz2 .
RUN tar xf openmpi-5.0.0.tar.bz2 \
    && cd openmpi-5.0.0 \
    && ./configure --prefix=$MPI_DIR \
    && make -j4 all \
    && make install \
    && cd .. && rm -rf \
    openmpi-5.0.0 openmpi-5.0.0.tar.bz2 /tmp/*

WORKDIR /home/ 
ENV PATH="$MPI_DIR/bin:/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="$MPI_DIR/lib:"

COPY ./ /home/pmmoto

CMD [ "tail","-f","/dev/null"]