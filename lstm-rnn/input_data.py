import numpy as np
import random
from sklearn.preprocessing import LabelBinarizer


class InputData:
    ALPHABETS = [letter for letter in 'abcdefghijklmnopqrstuvwxyz']
    FIXED_SIZE = 20     # a.k.a. MAX_TIME, batch_x[batch_count, FIXED_SIZE, len(ALPHABETS)]

    @classmethod
    def next_batch(cls, count=100, min_len=20, max_len=20):
        batch_x = []
        batch_seq_len_of_x = []
        batch_y = []
        for _ in xrange(count):
            seq_len = random.randint(min_len, max_len)
            word_vector = cls.next_word_vector(seq_len)
            seq_x = word_vector[0:-1]
            next_word_vector = word_vector[-1]
            batch_seq_len_of_x.append(seq_len)
            if seq_len < cls.FIXED_SIZE:
                seq_x = np.pad(seq_x, [(0, cls.FIXED_SIZE - seq_len), (0, 0)], 'constant')
            batch_x.append(seq_x)
            batch_y.append(next_word_vector)
        return np.array(batch_x), np.array(batch_seq_len_of_x), np.array(batch_y)

    @classmethod
    def next_word_vector(cls, step_count):
        WORD_TO_LEARN = 'hello'
        letters = ''
        for i in xrange(step_count + 1):
            letters += WORD_TO_LEARN[i % len(WORD_TO_LEARN)]
        binVectorList = cls.lettersToBinVectorList(letters)

        return binVectorList

    @classmethod
    def lettersToBinVectorList(cls, letters):
        indexes = cls.lettersToIndexes(letters)
        bins = []
        for index in indexes:
            bin = [0 for l in cls.ALPHABETS]
            bin[index] = 1
            bins.append(bin)
        bins = np.array(bins)
        if bins.shape[1] < len(cls.ALPHABETS):
            bins = np.pad(bins, [(0, 0), (0, len(cls.ALPHABETS) - bins.shape[1])], 'constant')
        return bins

    @classmethod
    def lettersToIndexes(cls, letters):
        return [cls.ALPHABETS.index(letter) for letter in letters]

    @classmethod
    def pad(cls, word_vector_list):
        if word_vector_list.shape[0] < cls.FIXED_SIZE:
            return np.pad(word_vector_list, [(0, cls.FIXED_SIZE - word_vector_list.shape[0]), (0, 0)], 'constant')
        else:
            return word_vector_list

if __name__ == '__main__':
    batch_x, batch_len, batch_y = InputData.next_batch(2, min_len=1, max_len=1)
    print(batch_x)
    print(batch_y)
