

from evaluator.evaluator import Evaluator
from models.rand_intent_parser import IntentParser
from models.rand_slot_filler import SlotFiller
from corpus.corpus_base import Corpus
from operator import itemgetter

class Pythia():
    def __init__(self, intent_set, slot_set):
        self.intent_set = intent_set
        self.slot_set = slot_set

    def predict(self, test_corpus):

        # First we predict slots
        slot_filler = SlotFiller()
        S = slot_filler.predict(test_corpus, self.slot_set)

        # Predict intents
        intent_parser = IntentParser()
        I = intent_parser.predict(test_corpus, self.intent_set)
        #print('predicted intents: ', I.keys())

        for i, s, x in zip(I, S, test_corpus):
            # intents with probabilities

            intents_with_p = {}
            for intent in self.intent_set:
                prob = i[intent]
                for slot in self.intent_set[intent]:
                    prob *= s[slot][1]
                intents_with_p[intent] = prob

            # Select the highest scoring intent
            predicted_intent = sorted(
                    intents_with_p.items(),
                    key=itemgetter(1),
                    reverse=True)[0][0]
            x.set_pred_intent(predicted_intent)

            # Select the slots of the predicted intent
            predicted_slots = {}
            for slot in self.intent_set[predicted_intent]:
                predicted_slots[slot] = s[slot][0] if slot in s else 'None' # if statement redundant?
            #print('\n==========PRED', predicted_slots)
            x.set_pred_slots(predicted_slots)


if __name__ == '__main__':
    corpus = Corpus(300, 'train')
    corpus.get_data()
    intent_set, slot_set = corpus.get_labels()

    model = Pythia(intent_set, slot_set)

    model.predict(corpus)

    eval = Evaluator(corpus, intent_set, slot_set)
    print(eval)

    #for x in corpus:
    #    print( '\n====')
    #    print( 'GOLD: ', x.get_gold_slots() )
    #    print( 'PRED: ', x.get_pred_slots() )