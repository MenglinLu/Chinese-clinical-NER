# coding = utf-8
import numpy as np
import jieba
import preprocess as p
import os
from keras.preprocessing.sequence import pad_sequences

# stop_word_file = 'dicts/stop_words.txt'
jieba.set_dictionary('data/dict.txt.big')
jieba.initialize()
word_embedding_file = 'data/word_embedding_matrix.npy'


def get_word_data(char_data):
    seq_data = [''.join(l) for l in char_data]
    word_data = []
    # stop_words = [line.strip() for line in open(stop_word_file, 'r', encoding='utf-8')]
    for seq in seq_data:
        seq_cut = jieba.cut(seq, cut_all=False)
        word_data.append([w for w in seq_cut for n in range(len(w))])

    return word_data

def get_word2object():
    word2vec = {}
    f = open(r'data/word2vec.bin') # load pre-trained word embedding
    i = 0
    for line in f:
        tep_list = line.split()
        if i == 0:
            n_word = int(tep_list[0])
            n_embed = int(tep_list[1])
        elif ord(tep_list[0][0]) > 122:
            word = tep_list[0]
            vec = np.asarray(tep_list[1:], dtype='float32')
            word2vec[word] = vec
        i += 1
    f.close()
    word2index = {k: i for i, k in enumerate(sorted(word2vec.keys()), 1)}
    return word2vec, n_word, n_embed, word2index

def get_word_embedding_matrix(word2vec, n_vocab, n_embed, word2index):
    embedding_mat = np.zeros([n_vocab, n_embed])
    for w, i in word2index.items():
        vec = word2vec.get(w)
        if len(vec) == n_embed:
            embedding_mat[i] = vec
    if not os.path.exists(word_embedding_file):
        np.save(word_embedding_file, embedding_mat)
    return embedding_mat

def add_word_data(word_data, word2index, max_length):
    index_data = []
    for l in word_data:
        index_data.append([word2index[s] if word2index.get(s) is not None else 0
                           for s in l])
    index_array = pad_sequences(index_data, maxlen=max_length, dtype='int32',
                                padding='post', truncating='post', value=0)
    return index_array


if __name__ == '__main__':
    char_train, tag_train = p.get_char_tag_data('data/train.txt')
    char_dev, tag_dev = p.get_char_tag_data('data/dev.txt')
    char_test, tag_test = p.get_char_tag_data('data/test.txt')
    # print(char_train[100][:20])

    word_train = get_word_data(char_train)
    word_dev = get_word_data(char_dev)
    word_test = get_word_data(char_test)
    # print(word_train[100][:20])

    word2vec, n_word, n_embed, word2index = get_word2object()
    n_vocab = len(word2vec.keys()) + 1
    print(n_word, n_vocab, n_embed) # 332648, 157142, 300

    if os.path.exists(word_embedding_file):
        word_embedding_matrix = np.load(word_embedding_file)
    else:
        word_embedding_matrix = get_word_embedding_matrix(word2vec, n_vocab,
                                                          n_embed, word2index)

    word_index_train = add_word_data(word_train, word2index, 200)
    word_index_dev = add_word_data(word_dev, word2index, 200)
    word_index_test = add_word_data(word_test, word2index, 200)
    print(word_index_train.shape, word_index_dev.shape, word_index_test.shape)
    # (21147, 200) (2362, 200) (4706, 200)

    np.save('data/word_train_add.npy', word_index_train)
    np.save('data/word_dev_add.npy', word_index_dev)
    np.save('data/word_test_add.npy', word_index_test)
