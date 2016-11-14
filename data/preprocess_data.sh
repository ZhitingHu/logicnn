# preprocess raw data
python preprocess_stsa.py ./raw/ ./w2v/GoogleNews-vectors-negative300.bin
# extract rule features
python logicnn_features.py ./stsa.binary.p 
