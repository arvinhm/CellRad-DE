# Stage 1: Build ROOT
FROM mcr.microsoft.com/windows/servercore:ltsc2022 AS root_build_stage

# Install required build tools (Visual Studio Build Tools for Windows)
RUN curl -SL --output vs_buildtools.exe https://aka.ms/vs/17/release/vs_buildtools.exe \
    && start /w vs_buildtools.exe --quiet --wait --norestart --nocache \
        --add Microsoft.VisualStudio.Workload.VCTools \
        --includeRecommended \
    && del /q vs_buildtools.exe

# Install other dependencies for ROOT (you may need to adjust depending on Windows compatibility)
RUN curl -o CMake.msi https://cmake.org/files/v3.22/cmake-3.22.1-windows-x86_64.msi \
    && msiexec /i CMake.msi /quiet /qn /norestart \
    && del CMake.msi

# Build ROOT (adjust the build to work with MSBuild instead of Unix tools)
RUN mkdir C:\root_build && cd C:\root_build && \
    curl -o root_v6.28.04.source.zip https://root.cern/download/root_v6.28.04.source.zip && \
    tar -xzf root_v6.28.04.source.zip && \
    mkdir root_build && cd root_build && \
    cmake ..\root-6.28.04 \
        -DCMAKE_INSTALL_PREFIX=C:\root \
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
        -DCMAKE_CXX_STANDARD=17 \
    && msbuild /p:Configuration=Release /maxcpucount /property:Platform=x64

# Stage 2: Build Geant4 and GATE on Windows
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Copy ROOT from the previous stage
COPY --from=root_build_stage C:\root C:\root

# Install Geant4 dependencies (adjust for Windows)
RUN curl -o Geant4.zip https://github.com/Geant4/geant4/archive/refs/tags/v11.2.1.zip \
    && tar -xzf Geant4.zip && \
    mkdir C:\geant4_build && cd C:\geant4_build && \
    cmake ..\geant4-11.2.1 \
        -DCMAKE_INSTALL_PREFIX=C:\geant4 \
        -DGEANT4_INSTALL_DATA=ON \
        -DGEANT4_USE_GDML=ON \
        -DGEANT4_USE_OPENGL_X11=OFF \
        -DGEANT4_USE_QT=OFF \
        -DGEANT4_BUILD_MULTITHREADED=ON \
        -DGEANT4_USE_SYSTEM_EXPAT=OFF \
        -DCMAKE_CXX_STANDARD=17 \
    && msbuild /p:Configuration=Release /maxcpucount /property:Platform=x64

# Build GATE (adjust paths and environment variables for Windows)
RUN curl -o Gate.zip https://github.com/OpenGATE/Gate/archive/refs/tags/v9.4.zip \
    && tar -xzf Gate.zip && \
    mkdir C:\gate_build && cd C:\gate_build && \
    cmake ..\Gate-9.4 \
        -DCMAKE_INSTALL_PREFIX=C:\gate \
        -DGeant4_DIR=C:\geant4 \
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
        -DGATE_USE_SYSTEM_EXPAT=OFF \
    && msbuild /p:Configuration=Release /maxcpucount /property:Platform=x64

# Set GATE environment variables
RUN setx PATH "%PATH%;C:\gate\bin"

# Set working directory
WORKDIR C:\APP

# Set command to run interactively
CMD ["cmd.exe"]
