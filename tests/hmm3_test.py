
import sys
import time
sys.path.append(sys.path[0] + '/../')
from models.hmm3_slot_filler_old import HMM3
from corpus.corpus_base import Corpus
from evaluator.evaluator import Evaluator

c = Corpus(2000, 'train')
intent_set, slot_set = c.get_labels()
c.shuffle()

m = HMM3(slot_set, intent_set)
m.train_model(c)

t = Corpus(100, 'test')
t.shuffle()

#start = time.time()

print('starting test')
#m.test_model(t)
#runtime = time.time() - start
#avg_runtime = round( (runtime / t.get_size()), 3)
#runtime = round(runtime, 3)

"""
for x in t:
    print('=' * 50)
    print('Text:\t[', x.get_utterance(), ']')
    print('Gold:\t[', x.get_gold_slots(), ']')
    print('Pred:\t[', x.get_pred_slots(), ']')
    print()
    slots_per_intent = x.get_pred_slots_per_intent()
    for i in slots_per_intent:
        print(i, ' (', round(slots_per_intent[i]['prob'], 3), ')\t[', slots_per_intent[i]['slots'])
"""

#e = Evaluator(t, intent_set, slot_set)
#print(e)

print('size slots: ', len(slot_set))

print('*' * 100)
_, test_slots = t.get_labels()

print('size slots: ', len(test_slots))