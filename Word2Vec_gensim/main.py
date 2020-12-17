import jieba
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import re


# 1
def cut(filename):
    lines = []
    punctuation = '[·!"#$%&\'()＃！（）*+,-./:;<=>?@，。：；?￥、…．【】［］《》〈〉『』「」？—－“”‘’[\\]^_`{|}~ ]+'
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = re.sub(punctuation, "", line)
            lines.append(list(jieba.cut(line)))
    return lines


if __name__ == '__main__':
    # 1
    sentences = cut("exp1_corpus.txt")
    print('Number of sentences: {}'.format(len(sentences)))
    # 2
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
    model.save('word2vec.model')
    # model.wv.save_word2vec_format('exp1.vector')
    # model = Word2Vec.load('word2vec300.model')

    # 3
    print(model.wv.similarity('中国', '中华'))
    print(model.wv.similarity('音乐会', '小提琴'))

    # 4
    print(model.wv.most_similar(positive=['武汉'], topn=5))
    print(model.wv.most_similar(positive=['人民'], topn=5))

    # 5
    print(model.wv.most_similar(positive=['湖北', '成都'], negative=['武汉'], topn=5))
    print(model.wv.most_similar(positive=['北约', '苏联'], negative=['美国'], topn=5))

    # 6
    l = ['江苏', '南京', '成都', '四川', '湖北', '武汉', '河南', '郑州', '甘肃', '兰州', '湖南', '长沙', '陕西', '西安', '吉林', '长春', '广东', '广州', '浙江', '杭州']
    embeddings = [model.wv[i] for i in l]
    pca = PCA(n_components=2)
    results = pca.fit_transform(embeddings)
    x = results[:, 0]
    y = results[:, 1]
    plt.figure(figsize=(15, 10))
    sns.set(font='Microsoft YaHei', font_scale=1.5)
    fig = sns.scatterplot(x=x, y=y)
    for i in range(len(l)):
        fig.text(x[i]+0.05, y[i], l[i])
    fig.get_figure().savefig('output.png')

