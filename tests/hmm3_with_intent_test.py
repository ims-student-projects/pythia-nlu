
import sys
import time
from operator import itemgetter
sys.path.append(sys.path[0] + '/../')
from models.hmm3_slot_filler import HMM3
from corpus.corpus_base import Corpus
from evaluator.evaluator import Evaluator

c = Corpus(2000, 'train')
intent_set, slot_set = c.get_labels()
c.shuffle()

m = HMM3(slot_set, intent_set)
m.train_model(c)

t = Corpus(10, 'test')
t.shuffle()

start = time.time()

m.test_model(t)
runtime = time.time() - start
avg_runtime = round( (runtime / t.get_size()), 3)
runtime = round(runtime, 3)


for x in t:
    print('=' * 50)
    print('Text:\t[', x.get_utterance(), ']')
    print('Gold:\t[', x.get_gold_slots(), ']')
    print('Pred:\t[', x.get_pred_slots(), ']')
    print()
    slots_per_intent = x.get_pred_slots_per_intent()
    intent_probs = {}
    for intent in slots_per_intent:
            intent_probs[intent] = slots_per_intent[intent]['prob']
    highest_intent = sorted(intent_probs.items(), key=itemgetter(1))[-1][0]
    print('Gold Intent: ', x.get_gold_intent())
    print('pred Intent: ', highest_intent)
    x.set_pred_intent(highest_intent)   
        


e = Evaluator(t, intent_set, slot_set)
print(e)


print(f'Total Runtime: {runtime} s. Avg for one instance: {avg_runtime} s.')
