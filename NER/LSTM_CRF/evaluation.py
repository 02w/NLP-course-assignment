def collect_entities_bio(sent, tags):

    append = lambda x, y, z: x.append('{}/{}'.format('_'.join(y), z.split('-')[1])) if len(y) != 0 else None

    type2label = {0: 'B-Loc', 1: 'I-Loc', 2: 'B-Org', 3: 'I-Org', 4: 'B-Peop', 5: 'I-Peop', 6: 'B-Other', 7: 'I-Other',
                  8: 'O', 9: '<START>', 10: '<END>'}
    entities = []
    entity = []
    # last_tag = vocab.T_START
    last_tag = '<START>'
    for word, tag in zip(sent, tags):
        if tag == 10:
            continue
        word = str(word)
        # tag = vocab.id_to_tag(tag)
        tag = type2label[tag]
        if tag == 'O':
            tag = 'O-O'
            if last_tag != tag:
                append(entities, entity, last_tag)
                entity = [word]
                last_tag = tag
            else:
                entity.append(word)
                last_tag = tag
        else:
            prefix, cls = tag.split('-')
            if prefix == 'B':
                append(entities, entity, last_tag)
                entity = [word]
                last_tag = tag
            else:
                if cls[-1] == last_tag[-1] and (last_tag[0] == 'B' or last_tag[0] == 'I'):
                    entity.append(word)
                    last_tag = tag
                else:
                    append(entities, entity, last_tag)
                    entity = [word]
                    last_tag = tag

    append(entities, entity, last_tag)

    return entities


def eval_metric(inputs, gold_labels, pred_labels, length):
    right = 0
    total = 0
    entity_gold_num = 0
    entity_pred_num = 0
    entity_right_num = 0

    for sent, gold_tags, pred_tags, l in zip(inputs, gold_labels, pred_labels, length):
        gold_tags = gold_tags[:l]
        pred_tags = pred_tags[:l]
        sent = sent[:l]
        pred_entities = collect_entities_bio(sent, pred_tags)
        gold_entities = collect_entities_bio(sent, gold_tags)
        # print(gold_entities)
        for gold_tag, pred_tag in zip(gold_tags, pred_tags):
            right += gold_tag == pred_tag
            total += 1

        for pred_entity in pred_entities:
            if pred_entity in gold_entities:
                entity_right_num += 1
        entity_gold_num += len(gold_entities)
        entity_pred_num += len(pred_entities)

    acc = right / total

    p = entity_right_num / entity_pred_num if entity_pred_num else 0
    r = entity_right_num / entity_gold_num if entity_gold_num else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return acc, p, r, f1


if __name__ == '__main__':
    sentence = [["Newspaper", "`", "Explains", "'", "U.S.", "Interests", "Section", "Events", "FL1402001894", "Havana", "Radio", "Reloj", "Network", "in", "Spanish", "2100", "GMT", "13", "Feb", "94"],
                ["`", "`", "If", "it", "does", "not", "snow", ",", "and", "a", "lot", ",", "within", "this", "month", "we", "will", "have", "no", "water", "to", "submerge", "150", ",", "000", "hectares", "(", "370", ",", "500", "acres", ")", "of", "rice", ",", "'", "'", "said", "Bruno", "Pusterla", ",", "a", "top", "official", "of", "the", "Italian", "Agricultural", "Confederation", "."]
                ]
    gold_labels = [[8, 8, 8, 8, 0, 8, 8, 8, 8, 0, 2, 3, 3, 8, 8, 6, 7, 6, 7, 7],
                   [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 7, 7, 7, 8, 6, 7, 7, 7, 8, 8,
                    6, 8, 8, 8, 8, 4, 5, 8, 8, 8, 8, 8, 8, 2, 3, 3, 8]]
    pred_labels = [[8, 8, 8, 8, 0, 8, 8, 8, 8, 0, 2, 3, 3, 8, 8, 6, 7, 6, 7, 7],
                   [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 7, 7, 7, 8, 6, 7, 7, 7, 8, 8, 6, 8, 8, 8, 8, 4, 5, 8, 8, 8, 8, 8, 8, 2, 3, 3, 8]]
    length = [19, 49]
    acc, p, r, f1 = eval_metric(sentence, gold_labels, pred_labels, length)
    print("acc={}".format(acc))
    print("f1={}".format(f1))
    print("p={}".format(p))