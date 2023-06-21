import sys
from pprint import pprint

sys.path.append('..')
from op.datasets import opp_tagged_acts

word_index = opp_tagged_acts.get_word_index()
labels = opp_tagged_acts.get_tags_labels()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def get_act_text(sequence):
    """Given a sequence of integers representing words,
    """
    return ' '.join([reverse_word_index.get(i-1, '') for i in sequence])


def get_act_tags(tags_ids):
    return [
        (
            labels[labels.id==t].iloc[0]['id'],
            labels[labels.id==t].iloc[0]['name'],
            labels[labels.id==t].iloc[0]['type']
        ) for t in tags_ids]

(train_data, train_labels), (test_data, test_labels) = opp_tagged_acts.load_data(num_words=None)

pprint(get_act_text(train_data[0]))
pprint(get_act_tags(train_labels[0]))