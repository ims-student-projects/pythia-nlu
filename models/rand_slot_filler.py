
import random
from operator import itemgetter
from nltk import ne_chunk, pos_tag, word_tokenize
from corpus.corpus_base import Corpus


class SlotFiller():

    def __init__(self):
        pass

    def predict(self, corpus, slots):
        predictions = []

        for x in corpus:
            # for each entity predict slot probabilities
            probs = {}
            entities = self.get_ne( x.get_utterance() )
            for e in entities:
                probs[e] = {}
                for s in slots:
                    probs[e][s] = random.uniform(0,1) 
                probs[e]['None'] = random.uniform(0,1)

            # for each slot, choose the entity with highest probability
            prediction = {}
            for s in slots:
                if s != 'None':
                    try:
                        highest = sorted([(e, probs[e][s]) for e in entities], key=itemgetter(1))[-1]
                        prediction[s] = highest
                    except:
                        prediction[s] = ('None', 0.0)
            predictions.append(prediction)
        return predictions
                


    def get_ne(self, txt):
        """
        Return a list of tokens & named entities from a string
        """
        tree = ne_chunk( pos_tag( word_tokenize( txt ) ) )
        return [n[0] if type(n)==tuple else ' '.join(e[0] for e in n) for n in tree]


if __name__ == '__main__':
    corpus = Corpus(5, 'train')
    corpus.get_data()
    intent_set, slot_set = corpus.get_labels()

    sf = SlotFiller()
    sf.predict(corpus, slot_set)