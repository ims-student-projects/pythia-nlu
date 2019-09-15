
import sys
sys.path.append(sys.path[0] + '/../')
from models.hmm3_slot_filler import HMM3
from corpus.corpus_base import Corpus
from evaluator.evaluator import Evaluator

c = Corpus(150, 'train')
intent_set, slot_set = c.get_labels()

m = HMM3(slot_set, intent_set)
m.train_model(c)

t = Corpus(3, 'test')
m.test_model(t)

e = Evaluator(t, intent_set, slot_set)
print(e)


for x in t:
    print('=' * 50)
    print('Text:\t[', x.get_utterance(), ']')
    print('Gold:\t[', x.get_gold_slots(), ']')
    print('Pred:\t[', x.get_pred_slots(), ']')
    print()
    slots_per_intent = x.get_pred_slots_per_intent()
    for i in slots_per_intent:
        print(i, ' (', round(slots_per_intent[i]['prob'], 3), ')\t[', slots_per_intent[i]['slots'])