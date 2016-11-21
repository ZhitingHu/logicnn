An implementation of the following paper in application of sentiment classification

[Harnessing Deep Neural Networks with Logic Rules](https://arxiv.org/abs/1603.06318)  
Zhiting Hu, Xuezhe Ma, Zhengzhong Liu, Eduard Hovy, Eric P. Xing, ACL 2016

  * The code is adapted from [previous work](https://github.com/yoonkim/CNN_sentence), written in Python (2.7) and Theano (0.9)

  * The [SST2](http://nlp.stanford.edu/sentiment/treebank.html) dataset is used

  * contact: zhitinghu@gmail.com

## Data Preparation ##

  * Download and uncompress the pre-trained [word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) to `data/w2v/`

  * run script `data/preprocess_data.sh` under `data/`

## Running ##

  * run `run.sh`

  * Average performance (accuracy) is around `q:0.893, p:0.887` (randomness comes from GPU parallelization)
