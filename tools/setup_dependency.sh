git submodule update

cd dependency

# Build OpenCV
cd opencv
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE ..
make -j10
cd ../..

# Build g2o
cd g2o
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j10
cd ../..

# Build Pangolin
cd Pangolin
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE ..
make -j10
cd ../..

# Download libtorch binary
wget -O libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.2.2.zip
unzip libtorch.zip 
rm libtorch.zip