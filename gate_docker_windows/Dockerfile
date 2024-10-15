# Stage 1: Build ROOT on AMD64
FROM amd64/ubuntu:22.04 AS root_build_stage

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required system packages for ROOT
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libssl-dev \
    python3 \
    python3-dev \
    libxml2-dev \
    libgl1-mesa-dev \
    libx11-dev \
    libxpm-dev \
    libxft-dev \
    libxext-dev \
    libjpeg-dev \
    libpng-dev \
    libldap2-dev \
    libfftw3-dev \
    libcfitsio-dev \
    libgsl-dev \
    libqt5opengl5-dev \
    libqt5svg5-dev \
    qtbase5-dev \
    qtwebengine5-dev \
    libcurl4-openssl-dev \
    libpq-dev \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    liblz4-dev \
    libzstd-dev \
    libtbb-dev \
    libtbb2 \
    libpcre3-dev \
    libpcre++-dev \
    libglew-dev \
    libmysqlclient-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxkbcommon-x11-dev \
    libxrender-dev \
    software-properties-common \
    curl

# Set up environment variables for library paths
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
ENV LIBRARY_PATH=/usr/local/lib:${LIBRARY_PATH:-}
ENV CPATH=/usr/local/include:${CPATH:-}

# Install ROOT from source
RUN mkdir /root_build && cd /root_build && \
    wget https://root.cern/download/root_v6.28.04.source.tar.gz && \
    tar -xzf root_v6.28.04.source.tar.gz && \
    mkdir root_build && cd root_build && \
    cmake ../root-6.28.04 \
        -DCMAKE_INSTALL_PREFIX=/usr/local/root \
        -Dbuiltin_freetype=ON \
        -Dbuiltin_pcre=ON \
        -Dbuiltin_zlib=ON \
        -Dminuit2=ON \
        -Droofit=ON \
        -Dssl=ON \
        -Dpython3=ON \
        -Dgdml=ON \
        -Dxrootd=OFF \
        -Dxml=ON \
        -DCMAKE_CXX_STANDARD=17 && \
    make -j$(nproc) && make install

# Stage 2: Build Geant4 and GATE on AMD64
FROM amd64/ubuntu:22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Copy ROOT from the previous stage
COPY --from=root_build_stage /usr/local/root /usr/local/root

# Update and install required system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libexpat1-dev \
    libxerces-c-dev \
    libx11-dev \
    libxmu-dev \
    libxpm-dev \
    libxt-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libmotif-dev \
    libfftw3-dev \
    libtbb-dev \
    libssl-dev \
    python3 \
    python3-pip \
    ca-certificates \
    libpng-dev \
    libjpeg-dev \
    libgsl-dev \
    libqt5opengl5-dev \
    libqt5svg5-dev \
    qtbase5-dev \
    qtwebengine5-dev \
    libcurl4-openssl-dev \
    gfortran \
    libldap2-dev \
    libxml2-dev \
    libkrb5-dev \
    libidn11-dev \
    libgfortran5 \
    libafterimage-dev \
    libpcre++-dev \
    liblzma-dev \
    libcfitsio-dev \
    libxext-dev \
    libxft-dev \
    libxinerama-dev \
    libxrandr-dev \
    libxkbcommon-x11-dev \
    libxrender-dev \
    libpq-dev \
    zlib1g-dev \
    libsqlite3-dev \
    libbz2-dev \
    libpcre3-dev \
    liblz4-dev \
    libzstd-dev \
    libtbb2 \
    software-properties-common \
    curl

# Set up environment variables for library paths
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/root/lib:${LD_LIBRARY_PATH:-}
ENV LIBRARY_PATH=/usr/local/lib:${LIBRARY_PATH:-}
ENV CPATH=/usr/local/include:${CPATH:-}

# Set ROOT environment variables
ENV ROOTSYS=/usr/local/root
ENV PATH=$ROOTSYS/bin:$PATH
RUN echo "source $ROOTSYS/bin/thisroot.sh" >> /etc/profile.d/root.sh

# Install Geant4 with multi-threading support (version 11.2.1)
RUN mkdir /geant4 && cd /geant4 && \
    wget https://github.com/Geant4/geant4/archive/refs/tags/v11.2.1.tar.gz && \
    tar -xzf v11.2.1.tar.gz && \
    mkdir geant4-build && cd geant4-build && \
    cmake ../geant4-11.2.1 \
        -DCMAKE_INSTALL_PREFIX=/usr/local/geant4 \
        -DGEANT4_INSTALL_DATA=ON \
        -DGEANT4_USE_GDML=ON \
        -DGEANT4_USE_OPENGL_X11=OFF \
        -DGEANT4_USE_QT=OFF \
        -DGEANT4_BUILD_MULTITHREADED=ON \
        -DGEANT4_USE_SYSTEM_EXPAT=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=17 && \
    make -j$(nproc) && make install

# Set Geant4 environment variables
RUN echo "source /usr/local/geant4/bin/geant4.sh" >> /etc/profile.d/geant4.sh
ENV LD_LIBRARY_PATH=/usr/local/geant4/lib:${LD_LIBRARY_PATH}

# Install GATE with multi-threading support
WORKDIR /gate
RUN git clone --branch v9.4 https://github.com/OpenGATE/Gate.git Gate
RUN mkdir gate-build
WORKDIR /gate/gate-build
RUN cmake ../Gate \
    -DCMAKE_INSTALL_PREFIX=/usr/local/gate \
    -DGeant4_DIR=/usr/local/geant4/lib/Geant4-11.2.1 \
    -DGATE_USE_OPTICAL=OFF \
    -DGATE_USE_DAVID=OFF \
    -DGATE_USE_ECAT7=OFF \
    -DGATE_USE_GPU=OFF \
    -DGATE_USE_ITK=OFF \
    -DGATE_USE_RTK=OFF \
    -DGATE_USE_UIVIS=OFF \
    -DGATE_USE_MPFR=OFF \
    -DGATE_USE_SYSTEM_CLHEP=OFF \
    -DGATE_USE_ANALYSIS=ON \
    -DGATE_USE_MULTI_THREAD=ON \
    -DGATE_USE_SYSTEM_EXPAT=OFF

# Build GATE with parallelization
RUN make -j$(nproc) && make install

# Set GATE environment variables
RUN echo "source /usr/local/gate/bin/gate_env.sh" >> /etc/profile.d/gate.sh
ENV PATH=/usr/local/gate/bin:$PATH

# Set working directory
WORKDIR /APP

# Set command to run interactively
CMD ["/bin/bash"]
