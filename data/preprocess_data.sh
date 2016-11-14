# preprocess raw data
python preprocess_stsa.py ./raw/ ../../data/word2vec/GoogleNews-vectors-negative300.bin
# extract rule features
python logicnn_features.py ./stsa.binary.p 
