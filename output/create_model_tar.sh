#!/usr/bin/env

# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

SERIALIZATION_DIR=$1 # eg. /absolute/path/to/keyphrase-generation/output/with/modelcheckpoints/
WEIGHTS_TH=$2  # for eg. best.th

MODEL_ARCHIVE=${SERIALIZATION_DIR}/model.tar.gz

if [ -f "${MODEL_ARCHIVE}" ]; then
    echo "Model archive exists: ${MODEL_ARCHIVE}"
    exit 1
fi

echo "Making model.tar.gz in ${MODEL_ARCHIVE}"

cd ${SERIALIZATION_DIR}

if [ -d "modeltar" ]; then
    echo "modeltar directory exists. Deleting ... "
    rm -r modeltar
fi

mkdir modeltar
cp -r vocabulary modeltar/
cp ${WEIGHTS_TH} modeltar/weights.th
cp config.json modeltar/
cd modeltar
tar -czvf model.tar.gz ./*
cp model.tar.gz ../
cd ../
rm -r modeltar