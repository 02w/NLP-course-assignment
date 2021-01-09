def collect_entities_bio(sent, tags, remove_o=False):

    def append(x, y, z):
        if remove_o:
            if z != 'O-O':
                x.append('{}/{}'.format('_'.join(y), z.split('-')[1])) if len(y) != 0 else None
        else:
            x.append('{}/{}'.format('_'.join(y), z.split('-')[1])) if len(y) != 0 else None

    entities = []
    entity = []

    last_tag = ' '
    for word, tag in zip(sent, tags):
        word = str(word)

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


def eval_metric(inputs, gold_labels, pred_labels, remove_o=False):
    right = 0
    total = 0
    entity_gold_num = 0
    entity_pred_num = 0
    entity_right_num = 0

    for sent, gold_tags, pred_tags in zip(inputs, gold_labels, pred_labels):
        sent = sent[1: -1]  # remove '<BOS>' and '<EOS>'
        pred_entities = collect_entities_bio(sent, pred_tags, remove_o)
        gold_entities = collect_entities_bio(sent, gold_tags, remove_o)

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
