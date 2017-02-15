import numpy as np
from sklearn.preprocessing import LabelBinarizer


class InputData:
    ALPHABETS = [letter for letter in 'ehlo']

    def __init__(self):
        pass

    @classmethod
    def next_batch(cls, size):
        WORD_TO_LEARN = 'hello'
        letters = ''
        for i in xrange(size):
            letters += WORD_TO_LEARN[i % len(WORD_TO_LEARN)]
        vectors = cls.lettersToVectors(letters)
        return vectors

    @classmethod
    def lettersToVectors(cls, letters):
        indexes = cls.lettersToIndexes(letters)
        binVectors = LabelBinarizer().fit_transform(indexes)
        xs = []
        ys = []
        for i in xrange(len(binVectors)):
            xs.append(binVectors[i])
            if i + 1 < len(binVectors):
                ys.append(binVectors[i + 1])
            else:
                ys.append(binVectors[0])
        return (np.array(xs), np.array(ys))

    @classmethod
    def lettersToIndexes(cls, letters):
        return [cls.ALPHABETS.index(letter) for letter in letters]

