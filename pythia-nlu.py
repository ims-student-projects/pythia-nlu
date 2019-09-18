
from math import log
from operator import itemgetter
from evaluator.evaluator import Evaluator
from models.baseline_svm_intent import SVM
from models.hmm3_slot_filler import HMM3
from corpus.corpus_base import Corpus


class Pythia():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def predict(self):

        intent_set, slot_set = self.train_data.get_labels()

        # train slot filler and intent parser on same dataset
        
        hmm = HMM3(slot_set, intent_set)
        hmm.train_model( self.train_data )
        hmm.test_model( self.test_data )

        svm = SVM( self.train_data, self.test_data )
        svm.train()
        # choose intent with highest prob as the one predicted 
        for x in self.test_data:
            intent_probs = x.get_intent_probabilities()
            predicted_intent = sorted(intent_probs.items(), key=itemgetter(1))[-1][0]
            x.set_pred_intent(predicted_intent)
        
        # report results
        print()
        print('=' * 50)
        print('SLOT FILLER AND INTENT PARSER TESTED SEPERATELY')
        e = Evaluator(self.test_data, intent_set, slot_set)
        print(e)

        # choose slots and intents with highest joint probability
        for x in self.test_data:
            joint_probs = {}
            for intent in intent_set:
                intent_prob = self.get_log_prob(x.get_intent_probabilities()[intent])
                slot_prob = x.get_pred_slots_per_intent()[intent]['prob']
                joint_probs[intent] = intent_prob + slot_prob
            highest_intent = sorted(joint_probs.items(), key=itemgetter(1))[-1][0]
            x.set_pred_intent(highest_intent)
            x.set_pred_slots( {'slots': x.get_pred_slots_per_intent()[highest_intent]['slots'], 'prob': 0})

        print()
        print('=' * 50)
        print('SLOT FILLER AND INTENT PARSER TESTED JOINT')
        e2 = Evaluator(self.test_data, intent_set, slot_set)
        print(e2)

        print('=' * 50)
        print()


    def get_log_prob(self, p):
        return -50 if p==0 else log(p)

if __name__ == '__main__':
    train = Corpus(70, 'train')
    train.shuffle()

    test = Corpus(20, 'test')
    test.shuffle()

    model = Pythia(train, test)

    model.predict()
