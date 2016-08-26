#!/bin/bash

echo 'preprocessing sst2 raw data ...'
python stsa_preprocessor.py sst/ word2vec/GoogleNews-vectors-negative300.bin

echo 'extracting features ...'
python feature_extractor.py ./stsa.binary.p 

echo 'data preprocessing done.'
