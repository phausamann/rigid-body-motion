#!/bin/sh

set -e

# Fix PY_VER to match the target version
PY_VER=`python -c 'import sys; version=sys.version_info[:3]; print("{0}.{1}".format(*version))'`

# liborocos-kdl
cd "${SRC_DIR}/orocos_kdl"
mkdir build && cd build

cmake \
  -D CMAKE_PREFIX_PATH=${PREFIX} \
  -D CMAKE_INSTALL_PREFIX="${PREFIX}" \
  -D CMAKE_BUILD_TYPE=Release \
  ..

make -j$CPU_COUNT
make install -j$CPU_COUNT


cd "${SRC_DIR}/python_orocos_kdl"
mkdir build && cd build

# PyKDL
cmake -D CMAKE_BUILD_TYPE=Release -D PYTHON_VERSION=3 ..

make -j$CPU_COUNT
cp PyKDL.so "${PREFIX}/lib/python${PY_VER}/site-packages"
