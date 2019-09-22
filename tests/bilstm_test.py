import sys
from operator import itemgetter

sys.path.append(sys.path[0] + '/../')
from models.baseline_svm_intent import SVM
from corpus.corpus_base import Corpus
from evaluator.evaluator import Evaluator
from models.bi_lstm import BiLSTM


train_corpus = Corpus(50, 'train')
intent_set, slot_set = train_corpus.get_labels()
train_corpus.shuffle()

test_corpus = Corpus(10, 'test')
test_corpus.shuffle()

model = BiLSTM(train_corpus, test_corpus)
model.train()
model.rnn_model()

# for x in test_corpus:
#     print()
#     print('=' * 50)
#     print('TEXT:\t[', x.get_utterance(), ']')
#     print('GOLD:\t', x.get_gold_intent() )
#     pred = x.get_intent_probabilities()
#     # chose intent with highest probability
#     intent = sorted( pred.items(), key=itemgetter(1) )[-1][0]
#     x.set_pred_intent(intent)
#     print('PRED:\t', intent )
#     print()

# e = Evaluator(test_corpus, intent_set, slot_set)
# print(e)