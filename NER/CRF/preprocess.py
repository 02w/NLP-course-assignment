import re


class Preprocess(object):
    def __init__(self, corpus_path, train=True):
        self.corpus_path = corpus_path
        self.train = train
        self.entity_maps = {'t': 'T', 'nr': 'PER', 'ns': 'LOC', 'nt': 'ORG'}
        self.bio2label = {'B-T': 0, 'I-T': 1, 'B-PER': 2, 'I-PER': 3, 'B-LOC': 4, 'I-LOC': 5, 'B-ORG': 6, 'I-ORG': 7,
                          'O': 8}
        self.raw_sentences = []
        self.character_lists = []
        self.tag_lists = []
        self.pos_lists = []
        self.label_lists = []
        self.word_gram = []
        self.feature = []

    def read_corpus(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split(' ')
                for w in words:
                    if w == '':
                        words.remove(w)
                if len(words[1:]) != 0:
                    self.raw_sentences.append(words[1:])

    def combine(self):
        # processed = []
        for i, words in enumerate(self.raw_sentences):
            processed = []
            index = 0
            temp = ''
            while True:
                word = words[index] if index < len(words) else ''
                if '[' in word:
                    temp += re.sub(pattern='/[a-zA-Z]*', repl='', string=word.replace('[', ''))
                elif ']' in word:
                    w = word.split(']')
                    temp += re.sub(pattern='/[a-zA-Z]*', repl='', string=w[0])
                    processed.append(temp + '/' + w[1])
                    temp = ''
                elif temp:
                    temp += re.sub(pattern='/[a-zA-Z]*', repl='', string=word)
                elif word:
                    processed.append(word)
                else:
                    break
                index += 1
            self.raw_sentences[i] = processed

    def get_tags(self):
        for i, words in enumerate(self.raw_sentences):
            c_list = []
            pos_list = []
            tag_list = []
            for word in words:
                w, pos = word.split('/')
                for c in w:
                    pos_list.append('n' if pos in self.entity_maps.keys() else pos)
                    c_list.append(c)
                    tag_list.append(self.entity_maps[pos] if pos in self.entity_maps.keys() else 'O')

            self.character_lists.append(['<BOS>'] + c_list + ['<EOS>'])
            self.pos_lists.append(['un'] + pos_list + ['un'])
            self.tag_lists.append(tag_list)

    def get_bio_label(self, numeric=False):
        for i, words in enumerate(self.tag_lists):
            last = ''
            label = []
            for c in words:
                if c == 'O':
                    label.append(self.bio2label[c] if numeric else c)
                elif c != last:
                    label.append(self.bio2label['B-' + c] if numeric else 'B-' + c)
                    last = c
                else:
                    label.append(self.bio2label['I-' + c] if numeric else 'I-' + c)
            self.label_lists.append(label)

    @staticmethod
    def segment_by_window(words_list, window=3):
        words = []
        begin, end = 0, window
        for _ in range(1, len(words_list)):
            if end > len(words_list):
                break
            words.append(words_list[begin: end])
            begin = begin + 1
            end = end + 1
        return words

    def extract_feature(self, word_grams):
        feature_list = []
        for index in range(len(word_grams)):
            for i in range(len(word_grams[index])):
                word_gram = word_grams[index][i]
                feature = {'w-1': word_gram[0], 'w': word_gram[1], 'w+1': word_gram[2],
                           'w-1:w': word_gram[0] + word_gram[1], 'w:w+1': word_gram[1] + word_gram[2],
                           # 'p-1': self.pos_lists[index][i], 'p': self.pos_lists[index][i + 1],
                           # 'p+1': self.pos_lists[index][i + 2],
                           # 'p-1:p': self.pos_lists[index][i] + self.pos_lists[index][i + 1],
                           # 'p:p+1': self.pos_lists[index][i + 1] + self.pos_lists[index][i + 2],
                           'bias': 1.0}
                feature_list.append(feature)
            self.feature.append(feature_list)
            feature_list = []
        # return features

    def process(self):
        if self.train:
            self.read_corpus()
            self.combine()
            self.get_tags()
            self.get_bio_label()
        else:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.character_lists.append(['<BOS>'] + list(line.strip()) + ['<EOS>'])
        self.word_gram = [self.segment_by_window(w) for w in self.character_lists]
        # self.feature = self.extract_feature(self.word_gram)
        self.extract_feature(self.word_gram)
