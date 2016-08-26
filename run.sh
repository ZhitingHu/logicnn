for i in 1 2 3 4 5
do
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python logicnn_sentiment_bak.py -nonstatic -word2vec
done
