FROM ubuntu:18.04
WORKDIR /root
ADD display_x11.patch /root/dependencies/display_x11.patch
RUN apt-get update \
# CMake
&& apt-get -y install cmake \
# Eigen3
&& apt-get -y install libeigen3-dev \
# OpenCV 3.1
&& cd /root/dependencies \
&& apt-get -y install wget unzip libgtk2.0-dev pkg-config \
&& wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip \
&& unzip opencv.zip \
&& cd opencv-3.1.0 \
&& mkdir build \
&& cd build \
&& cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_FFMPEG=OFF .. \
&& make -j4 \
&& make install \
&& cd /root/dependencies \
# Sophus
&& apt-get -y install git \
&& git clone https://github.com/strasdat/Sophus.git \
&& cd Sophus \
&& git checkout 13fb328 \
&& mkdir build \
&& cd build \
&& cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local .. \
&& make -j4 \
&& make install \
&& cd /root/dependencies \
# G2O
&& apt-get -y install qt5-qmake qt5-default libqglviewer-dev-qt5 libsuitesparse-dev libcxsparse3 libcholmod3 \
&& git clone https://github.com/RainerKuemmerle/g2o.git \
&& cd g2o \
&& git checkout 9b41a4e \
&& mkdir build \
&& cd build \
&& cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local .. \
&& make -j4 \
&& make install \
&& cd /root/dependencies \
# Pangolin
&& apt-get -y install libgl1-mesa-dev libglew-dev freeglut3-dev \
&& git clone https://github.com/stevenlovegrove/Pangolin.git \
&& cd Pangolin \
&& git checkout 1ec721d \
&& patch src/display/device/display_x11.cpp ../display_x11.patch \
&& mkdir build \
&& cd build \
&& cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local .. \
&& make -j4 \
&& make install \
&& cd /root \
&& rm -r dependencies \
# This Repo
&& apt-get -y install libboost-dev \
&& git clone https://github.com/BerwinZ/RGBD_VisualOdometry.git \
&& cd RGBD_VisualOdometry \
&& mkdir build \
&& cd build \
&& cmake .. \
&& make -j4 \
&& cd /root \
# Extra
&& apt-get -y install vim
CMD cd && /bin/bash