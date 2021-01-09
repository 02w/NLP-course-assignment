from preprocess import Preprocess
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split
import joblib
from evaluation import eval_metric, collect_entities_bio
from collections import Counter


if __name__ == '__main__':
    data = Preprocess('rmrb.txt')
    data.process()

    ner = CRF()
    # ner = joblib.load('model-no-pos.model')

    X_train, X_test, y_train, y_test, sent_train, sent_test = train_test_split(data.feature, data.label_lists, data.character_lists, test_size=0.3, random_state=20)

    ner.fit(X_train, y_train)
    joblib.dump(ner, 'model-no-pos.model')

    y_pred = ner.predict(X_test)

    acc, p, r, f1 = eval_metric(sent_test, y_test, y_pred, remove_o=True)
    print(f'Acc: {acc}    Precision: {p}    Recall: {r}     f1: {f1}')

    labels = ner.classes_
    labels.remove('O')
    print(flat_classification_report(y_test, y_pred, labels=labels))

    sentence = Preprocess('test.txt', train=False)
    sentence.process()
    pred = ner.predict(sentence.feature)
    for i, s in enumerate(sentence.character_lists):
        print(collect_entities_bio(s[1: -1], pred[i], remove_o=True))

    def print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


    print("Top likely transitions:")
    print_transitions(Counter(ner.transition_features_).most_common(10))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(ner.transition_features_).most_common()[-10:])
