#!/usr/bin/env bash

PKG_NAME=rigid-body-motion
USER=phausamann

anaconda -t $ANACONDA_TOKEN upload -u $USER $CONDA_BLD_PATH/noarch/$PKG_NAME*.tar.bz2 --force
