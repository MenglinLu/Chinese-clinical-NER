import ahocorasick
from typing import List


class ACA(object):
    def __init__(self):
        self.ac = ahocorasick.Automaton()

    def add_words(self, words: List[str]):
        [self.ac.add_word(w, w) for w in words]
        self.ac.make_automaton()

    def get_hits(self, text) -> List[str]:
        return list(set([r[1] for r in list(self.ac.iter(text))]))

    '''
    'iii度烧伤46%,检测了abo血型是a型,检测了murphy征是+++,bp135/66mmhg'
    [(5, 'iii度烧伤'), (17, 'abo血型'), (28, 'p'), (31, 'murphy征'), (38, 'p')]
    
    the first element is the last index, the second element is the matching name
    need to find the first index of the matching name
    '''
    def get_hits_with_index(self, text, sorted_by_index=False) -> List:
        result = list(self.ac.iter(text))
        # sort the result first by index ascending, if tie, sort the index name descending
        if sorted_by_index:
            result = sorted(result, key=lambda x: (x[0]-len(x[1]), -1*len(x[1])), reverse=False)
        return result
