## NLP Course assignments
### NER
- CRF: use `sklearn-srfsuite` and `rmrb1998` dataset
- BiLSTM: `conll04` dataset
- BiLSTM-CRF: `conll04` dataset, reference: [pytorch tutorials](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html) / [batch_bilstm_crf.py](https://github.com/liaozhihui/MY_MODEL/blob/master/batch_bilstm_crf.py) 

### Sentiment Classification
- BiLSTM and TextCNN

### Sentiment Classification with pretrained wordvectors
- BiLSTM + Attention
- Pretrained word vector from [Embedding/Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)

Note: `requires_grad_` of `nn.Embedding` is set to `True`, which may be incorrect.

### Word2Vec_gensim
- use `gensim` to train Word2Vec
