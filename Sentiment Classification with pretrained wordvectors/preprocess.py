import jieba
from gensim.models import Word2Vec


def cut(filename):
    lines = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.append(list(jieba.cut(line[2:])))
    return lines[1:]


if __name__ == '__main__':
    # 1
    sentences = cut('dataset/train.csv')
    print('Number of sentences: {}'.format(len(sentences)))
    # 2
    model = Word2Vec(sentences, size=300, window=5, min_count=1, workers=4)
    # model.save('word2vec.model')
    model.wv.save_word2vec_format('gensim300.vector')
    # model = Word2Vec.load('word2vec300.model')


