from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import numpy as np


def _preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def _is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def _clean_seq(seq):
    processed_seq = []
    # Convert to lower case
    seq = seq.lower()
    # Replaces URLs with the word URL
    seq = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', seq)
    # Replace 2+ dots with space
    seq = re.sub(r'\.{2,}', ' ', seq)
    # Strip space, " and ' from seq
    seq = seq.strip(' "\'')
    # Replace multiple spaces with a single space
    seq = re.sub(r'\s+', ' ', seq)
    words = seq.split()

    for word in words:
        word = _preprocess_word(word)
        if _is_valid_word(word):
            processed_seq.append(word)

    return ' '.join(processed_seq)


def preprocess_comments(corpus, test=False):
    preprocessed_corpus = []
    file_to_save = "data/processed-comments.csv"
    with open(file_to_save, "w", encoding="utf-8") as csv:
        print("Preprocessing corpus")
        total = len(corpus)
        for i, sentence in enumerate(corpus):
            sentence = _clean_seq(str(sentence))
            preprocessed_corpus.append(sentence)
            csv.write("%s,\n" % sentence)
            write_status(i, total)
    csv.close()

    print("\nPreprocessing complete")
    return preprocessed_corpus


def next_batch(num, data, labels):
    '''
	Return a total of `num` random samples and labels.
	'''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:, :num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def write_status(i, total):
    sys.stdout.write('\r')
    sys.stdout.write('Processing %d/%d' % (i, total))
    sys.stdout.flush()
