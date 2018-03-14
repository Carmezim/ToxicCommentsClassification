import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import csv
import pickle
import pprint
import json
import os
import sys

from tensorflow.contrib.keras import preprocessing
from .preprocess import preprocess_comments


class Files:
    """ Datasets and other relevant files. """
    submission = "data/test.csv"
    train = "data/train.csv"
    test = "data/processed-test.csv"
    train_processed = "data/preprocessed-train.csv"
    glove = "data/glove.6B.300d.txt"
    numbersbatch = "data/numberbatch-en-17.06.txt"


class Config:
    b = 128  # batch size
    ckp_dir = "checkpoints/"
    model_name = "check.ckpt"
    i = 100000  # number of iterations
    i_p = .45  # probability input will be retained by dropout
    lr = 0.1  # learning rate
    lr_decay = 0.8  # learning rate decay rate
    lr_decay_steps = 10000  # learning rate decay steps
    max_grad = 5  # max value gradient should reach (for clipping)
    max_len = 4000  # max sequence length
    max_to_keep = 10  # max checkpoints to keep
    n_classes = 6  # number of classes to predict
    n_dim = 300  # word vector dimension
    n_epochs = 10  # number of epochs
    n_hidden = 256  # number of lstm units
    o_p = .6  # probability output will be retained by dropout
    summary_dir = "summary/"
    v = 80000  # vocabulary size
    w_d = 1e-3  # weight decay


def make_dir(path: str) -> None:
    """ Creates directory. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def check_file(filepath: str) -> bool:
    """ Checks for existing file. """
    try:
        return os.path.isfile(filepath)
    except OSError:
        print("File does not exist")


def check_data(data) -> None:
    """ Displays first samples of data. """
    print(data.head())


def _load_pickle(filename: str, data_to_dump=None) -> bytearray:
    """ Check if pickle file exists and load file if it does. If no file is
        found create it.

        Args:
            filepath: directory of pickle file to be loaded.

        Returns:
            file: loaded pickle file.
    """
    file_dir = "data/pickles/%s" % str(filename)
    if check_file(file_dir):
        print("%s file found, loading..." % filename)
        with open(file_dir, "rb") as f:
            file = pickle.load(f)
            f.close()
        return file

    elif data_to_dump is not None:
        print("%s file not found, creating..." % filename)
        with open(file_dir, "wb") as f:
            pickle.dump(data_to_dump, f, pickle.HIGHEST_PROTOCOL)
            f.close()


def _load_data_helper(file_name) -> pd.DataFrame:
    file_name = str(file_name)
    data = None
    if check_file("data/processed-%s" % file_name):
        print("Processed %s data found" % file_name)
        data = pd.read_csv(str("data/processed-%s" % file_name), header=0,
                           encoding="utf-8", low_memory=False)
    return data


def _preprocess_data(file, inp_header=None, cls=None, test=False) -> \
        pd.DataFrame:
    print("Processed data not found, loading and pre-processing...")

    data = pd.read_csv("data/" + str(file), header=0, encoding="utf-8",
                       low_memory=False)
    check_data(data)
    corpus = data["comment_text"].fillna("_na_").values
    ids = data["id"].values
    corpus_processed = preprocess_comments(corpus)

    if not test:
        # preprocess training data then save locally.
        file_path = "data/processed-%s" % str(file)
        labels = data[cls].values
        del data
        # create dataframe with processed data and save to file
        p_data = pd.DataFrame(labels, columns=cls)
        p_data["id"] = ids
        p_data["comment_text"] = corpus_processed
        p_data = p_data[inp_header + cls]
        p_data.head()
        p_data.to_csv(file_path, sep=",", encoding="utf-8", index=False)

        print("processed data sample:")
        p_data.head()

    else:
        del data
        # preprocess test data saving it locally.
        file_path = "data/processed-%s" % str(file)
        p_data = _load_data_helper(str(file))
        if not p_data:
            p_data = pd.DataFrame(columns=inp_header)
            p_data['id'] = ids
            p_data['comment_text'] = corpus_processed
            p_data.to_csv(file_path, sep=",", encoding="utf-8", index=False)
            print("\nTest data sample", p_data.head())

    print("Processed data saved to processed-%s" % file)
    return p_data


def write_status(i: int, total: int) -> None:
    """ Writes status of a process to console.

        Args:
            i: index of the current iteration.
            total: number of iterations.
    """
    sys.stdout.write('\r')
    sys.stdout.write('Processing %d/%d' % (i, total))
    sys.stdout.flush()


def _split_validation(train_data=None, labels=None, percentage=0.9) -> object:
    """ Splits train data into train and validation.

        Args:
            train_data: training corpus
            labels: training data labels
            percentage: percentage to which the training data will be split
            into training and validation defaulting to 90% training and 10%
            validation.
        Returns:
            train_X, train_Y, val_X, val_Y, train_data, labels:
            split training and validation data with respective labels and
            provided corpus with its labels.
    """
    print("Splitting data into training and validation...")
    data_size = len(train_data)
    # index to be split the data at
    s_idx = int(percentage * data_size)
    train_X, train_Y = train_data[:s_idx], labels[:s_idx]
    val_X, val_Y = train_data[s_idx:], labels[s_idx:]

    return train_X, train_Y, val_X, val_Y, train_data, labels


def _load_data(data_file, split_percentage=0.9, test=False):
    """ Loads datasets and perform simple pre-processing handling
        missing samples.

        Args:
            data_file: CSV file directory which is composed of ID, TEXT and
            LABELS.
            split_percentage: percentage to which to split train data
            into train and validation, defaulted to 90% train dn 10%
            validation.
        Returns:
            split_validation(): properly splits and returns the data.
    """
    file = data_file.split("-")[1]
    input_header = ["id", "comment_text"]
    classes = ["toxic",
               "severe_toxic",
               "obscene",
               "threat",
               "insult",
               "identity_hate"]
    columns = input_header + classes

    if not test:
        print("\nLooking for existing processed data...")
        data = _load_data_helper(file)
        if data is not None:
            print("Training data head")
            check_data(data)
            corpus_processed = data["comment_text"].values
            labels = data[classes].values
            return _split_validation(corpus_processed.astype(str),
                                     labels, split_percentage)
        else:
            processed_data = _preprocess_data(file, input_header, classes, False)

            return _split_validation(
                processed_data["comment_text"].values.astype(str),
                processed_data[classes].values, split_percentage)

    else:
        print("Loading test data...")
        print("Looking for existing processed test data...")

        test_data = _load_data_helper(file)
        if test_data is not None:
            return test_data

        else:
            processed_data = _preprocess_data(file, input_header, None, True)
            print("processed data returned")
            return processed_data


def _load_vocab(sentences, vocab_size) -> dict:
    """ Generates the corpus vocabulary.

        Args:
            sentences: list of the corpus comments.
            vocab_size: the size of vocabulary.
        Returns:
            vocab: vocabulary in a dict formed by {word: index}.
    """
    file_name = "data/vocab-%s.json" % str(vocab_size)
    print("Looking for existing vocabulary...")

    try:
        vocab_file = open(file_name, 'r')
        vocab = json.load(vocab_file)
        print("Vocabulary found and loaded")

    except Exception as e:
        vocab_file = open(file_name, 'w')
        print(str(e))
        print("No existing vocabulary found, generating new vocab...")
        all_words = []
        for sentence in sentences:
            words = sentence.split()
            all_words.extend(words)

        unique_words = list(set(all_words))
        vocab = {word: i for i, word in enumerate(unique_words) \
                 if i <= vocab_size}
        print("Vocabulary created. Saving...")
        json.dump(vocab, vocab_file)
        print("Vocabulary saved")

    vocab_file.close()
    print("returning vocab")
    return vocab


def _load_glove_vectors(vocab, glove_file):
    """ Loads glove embeddings and for each word in the vocabulary assigns
        the corresponding vector representation if word exists on glove vectors.

        Args:
            vocab: corpus vocabulary (list) -> confirm type.
            glove_file: glove vectors directory. The file is organized as
            [word, vectors] in each line.
        Returns:
            embeddings: word embeddings for vocabulary (dict {word:vectors}).
    """
    print("Looking for pre-loaded glove vectors...")
    glove_embeddings = _load_pickle("glove.pkl")
    found = 0

    if glove_embeddings:
        print("Glove vectors found.\n")
        return glove_embeddings

    else:
        print("\nSaved vocab glove vectors not found, loading pre-trained " +
              "word vectors...")
        glove_embeddings = {}

    with open(glove_file, "r", encoding="utf-8") as glove_file:
        # locate corresponding glove pre-trained word vector
        # for each instance in the vocabulary if existent
        for i, line in enumerate(glove_file):
            write_status(i + 1, 0)
            tokens = line.split()
            word = tokens[0]
            # for each word in the glove embeddings,
            # if it's also present in the vocabulary, get pre-trained
            # word vector returning a dict of word:vector (word2vec)
            if vocab.get(word):
                word_vector = [float(o) for o in tokens[1:]]
                glove_embeddings[word] = np.array(word_vector)
                found += 1

        glove_file.close()
    print("\n%d words from corpus found in glove" % (found - 1))
    print("%d words not found" % (len(vocab) - found))
    print("Pre-trained vectors loaded")
    print("Saving vocabulary glove pre-trained vectors...\n")
    _load_pickle("glove.pkl", glove_embeddings)
    return glove_embeddings


def _load_cn_vectors(vocab, nb_file):
    """ Loads ConceptNet embeddings and for each word in the vocabulary assigns
        the corresponding vector representation if word exists on glove vectors.

        Args:
            vocab: corpus vocabulary (list) -> confirm type.
            nb_file: glove vectors directory. The file is organized as
            [word, vectors] in each line.
        Returns:
            cn_embeddings: word embeddings for vocabulary (dict {word:vectors}).
    """
    print("Looking for pre-loaded ConceptNet vectors...")
    cn_embeddings = _load_pickle("cn.pkl")
    found = 0

    if cn_embeddings:
        print("ConceptNet vectors found.\n")
        return cn_embeddings

    else:
        print("\nConceptNet vectors not found, loading pre-trained " +
              "word vectors...")
        cn_embeddings = {}

    with open(nb_file, "r", encoding="utf-8") as glove_file:
        # locate corresponding glove pre-trained word vector
        # for each instance in the vocabulary if existent
        for i, line in enumerate(glove_file):
            write_status(i + 1, 0)
            tokens = line.split()
            word = tokens[0]
            # for each word in the glove embeddings,
            # if it's also present in the vocabulary, get pre-trained
            # word vector returning a dict of word:vector (word2vec)
            if vocab.get(word):
                word_vector = [float(o) for o in tokens[1:]]
                cn_embeddings[word] = np.array(word_vector)
                found += 1

        glove_file.close()
    print("\n%d words from corpus found in ConceptNet" % (found - 1))
    print("%d words not found" % (len(vocab) - found))
    print("Pre-trained vectors loaded")
    print("Saving vocabulary ConceptNet pre-trained vectors...\n")
    _load_pickle("cn.pkl", cn_embeddings)
    return cn_embeddings


def _feature_vector(sentence, vocab):
    """ Creates each word vector for each sentence.

        Args:
            sentence: a sentence from the corpus.
            vocab: vocabulary.
        Returns:
            feature_vector: vector representation from sentence.
    """
    words = sentence.split()
    feature_vector = []
    # get indices for each word in the vocabulary
    for i in range(len(words) - 1):
        word = words[i]
        if vocab.get(word) is not None:
            feature_vector.append(vocab.get(word))
        # else:
        #     feature_vector.append(399999)  # unknown word vector
    if len(words) >= 1:
        if vocab.get(words[-1]) is not None:
            feature_vector.append(vocab.get(words[-1]))
    return feature_vector


def _generate_feature_vectors(sentences=None, sentiments=None, vocab=None,
                              test_data=False):
    """ Creates feature vectors for each sentence in the corpus (word to index).

        Args:
            sentences: list of each sentence in the corpus.
            sentiments: labels for each sentence.
            vocab: vocabulary.
            test_data: flag to differentiate between training and test sets.
        Returns:
            vectors: list of each sentence vectorized (feature
            vectors for all words in each sentence).
            labels: np.ndarray shape (N_samples, N_classes) of labels for each
            sentence feature vector.
    """
    total = len(sentences)

    if not test_data:
        vectors_file = "feature_vectors-%s.pkl" % total
        labels_file = "labels-%s.pkl" % total

        print("Looking for feature vectors and labels...")
        vectors = _load_pickle(vectors_file)
        labels = _load_pickle(labels_file)

        if vectors and labels is not None:
            return vectors, np.array(labels)
        else:
            vectors = []
            labels = []
            print("Files not found, generating feature vectors...")
            for i in range(0, total):
                # create vector of words indices from each comment
                sentence_vectors = _feature_vector(sentences[i], vocab)
                vectors.append(sentence_vectors)
                labels.append(sentiments[i])
                write_status(i, total)

            # Save data to pickle
            _load_pickle(vectors_file, vectors)
            _load_pickle(labels_file, np.array(labels))
            print("Feature vectors created and saved.")
            return vectors, np.array(labels)

    else:
        vectors_file = "feature_vectors-test-%s.pkl" % Config.v
        print("Looking for test data feature vectors...")
        vectors = _load_pickle(vectors_file)
        if vectors:
            print("Test data feature vectors found.")
            return vectors
        else:
            vectors = []
            print("Test data feature vectors not found, generating...")
            for i, sentence in enumerate(sentences):
                feature_vector = _feature_vector(sentence, vocab)
                vectors.append(feature_vector)
                write_status(i, total)

            print("\nFeature vectors created")
            print("Saving vectors to file")
            _load_pickle(vectors_file, vectors)

            return vectors


def _embedding_matrix(glove_embeds, vocab, cfg):
    """ Creates embedding matrix initializing embedding matrix randomly for
        words in the corpus that aren't present in glove pre-trained word
        vectors then fetching the ones available.

        Args:
            glove_embeds: vocabulary embeddings with glove vectors.
            vocab: vocabulary (dict).
            cfg: object containing hyperparameters.
        Returns:
            embed_matrix: np.ndarray matrix of all words' indices mapped to
            their vector representation, e.g. an embedding matrix of shape (
            vocab_size. word_vector_dimension) defined as [vocab_indices,
            vocab_vectors].
    """
    print("Looking for embedding matrix...")
    embedding_file = "embed_matrix.pkl"
    total = len(vocab)
    embeds = _load_pickle(embedding_file)

    if embeds is not None:
        return embeds

    else:
        print("Embedding matrix not found, creating...")
        embed_matrix = np.random.randn(cfg.v + 1, cfg.n_dim) * .01
        for word, i in vocab.items():
            vector = glove_embeds.get(word)
            if vector is not None:
                # maps each word vector to  word's index
                # embed_matrix == word_index:word_vector
                embed_matrix[i] = vector
            else:
                # if vector not present create a random one
                embed_matrix[i] = np.array(
                    np.random.uniform(-1., 1., cfg.n_dim))
            write_status(i, total)
        assert embed_matrix.shape == (cfg.v + 1, cfg.n_dim)
        print("\nEmbedding matrix created.")
        print("Saving embedding matrix locally...")
        _load_pickle(embedding_file, embed_matrix)
        print("\nEmbeddings saved.")
        return embed_matrix


def minibatches(inputs, labels, batch_size: int, shuffle=False):
    """ Generate minibatches from data.
        reference: https://stackoverflow.com/questions/38157972/\
        how-to-implement-mini-batch-gradient-descent-in-python

        Args:
            inputs: input data.
            labels: data labels.
            batch_size: size of the minibatches.
            shuffle: shuffle flag default to False.
        Yields:
            inputs[batch_indices]: input batch.
            labels[batch_indices]: labels batch.
    """
    inputs = np.asarray(inputs)
    size = inputs.shape[0]
    assert inputs.shape[0] == labels.shape[0]
    labels = np.asarray(labels, dtype=np.int)
    indices = np.arange(size)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in np.arange(0, size - batch_size + 1, batch_size):
        batch_indices = slice(start_idx, start_idx + batch_size)
        # if inputs[batch_indices].shape[0] < batch_size:
        yield inputs[batch_indices], labels[batch_indices]


def test_data_vecs(f, c):
    vecs_file = "feature-vectors-test-%s.pkl" % c.v
    embeds_test_file = "embed-matrix-test-%s.pkl" % c.v

    vecs = _load_pickle(vecs_file)
    embed_matrix = _load_pickle(embeds_test_file)

    if vecs or embed_matrix is None:
        data = _load_data(f.test, test=True)
        corpus = data["comment_text"].fillna("__na__").values
        print("corpus sample", corpus[:10])
        vocab = _load_vocab(corpus, c.v)
        vecs = _generate_feature_vectors(corpus, None, vocab, True)
        embed_matrix = _embedding_matrix(vecs, vocab, c)
        _load_pickle(embeds_test_file, embed_matrix)
        _load_pickle(vecs_file, vecs)
        del data

    assert type(vecs) == list
    assert type(embed_matrix) == np.ndarray

    return vecs, embed_matrix


def return_data(test=False):
    files = Files()
    c = Config()

    if not test:
        # retrieving data
        X, Y, val_X, val_Y, _, _ = _load_data(files.train_processed)

        vocab = _load_vocab(X, c.v)
        cn_vectors = _load_cn_vectors(vocab, files.numbersbatch)
        feature_vectors, labels = _generate_feature_vectors(X, Y, vocab)
        embed_matrix = _embedding_matrix(cn_vectors, vocab, c)
        # validation data
        val_feat_vecs, val_labels = _generate_feature_vectors(val_X, val_Y,
                                                              vocab)

        feature_vectors = preprocessing.sequence.pad_sequences(feature_vectors,
                                                               value=1,
                                                               maxlen=c.max_len,
                                                               padding="post")

        val_feat_vecs = preprocessing.sequence.pad_sequences(val_feat_vecs,
                                                             value=1,
                                                             maxlen=c.max_len,
                                                             padding="post")

        return [feature_vectors, labels, embed_matrix, val_feat_vecs,
                val_labels]

    else:
        # retrieve test data feature vectors
        test_feat_vecs, test_embed_matrix = test_data_vecs(files, c)

        # pad to defined dimensions
        padded_test_vecs = preprocessing.sequence.pad_sequences(test_feat_vecs,
                                                                value=1,
                                                                maxlen=c.max_len,
                                                                padding="post")

        return [padded_test_vecs, test_embed_matrix]


def pad_vectors(vecs, shape):
    res = np.zeros(shape)
    slices = [slice(0, min(dim, shape[e])) for e, dim in enumerate(vecs.shape)]
    res[slices] = vecs[slices]
    return res
