
import sys
sys.path.append(sys.path[0] + '/../')
from models.hmm3_slot_filler import HMM3
from corpus.corpus_base import Corpus


c = Corpus(50, 'train')
intent_set, slot_set = c.get_labels()

m = HMM3(slot_set, intent_set)
m.train_model(c)

t = Corpus(1, 'test')
m.test_model(t)
