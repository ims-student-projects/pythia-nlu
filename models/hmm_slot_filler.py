
import numpy as np
from math import log2
from nltk.tokenize import word_tokenize as tokenize
from progress.bar import PixelBar
from corpus.corpus_base import Corpus

class HMM_Slot_Filler():

    def __init__(self, slot_set, intent_set):

        # Set of slot labels in IOB format
        self.slot_set = slot_set
        self.intent_set = intent_set
        self.S = self.extract_iob_labels(slot_set)


    def train_model(self, train_data):

        print('Traininig HMM-3 model...')
        corpus_size = 0

        # Create a hash with index numbers assigned to each slot label and sentence word
        self.label2index = { label:i for label, i in zip( self.S, range(len(self.S)) ) }
        self.label2index['<START>'] = len(self.label2index)
        self.label2index['<STOP>'] = len(self.label2index)
        self.word2index = {}
        i = 0
        for x in train_data:
            corpus_size += 1
            for w in tokenize( x.get_utterance() ):
                if w not in self.word2index:
                    self.word2index[w] = i
                    i += 1

        print('Initializing probability matrixes...')

        # Define probability and count matrixes
        k = len(self.label2index)
        n = len(self.word2index)
        self.emit_p = np.zeros( shape=(n, k-2) )
        self.trans_p = np.zeros( shape=(k-1, k-1, k) )

        unigram_c = np.zeros( shape=(k) )
        bigram_c = np.zeros( shape=(k-1, k) )

        print('Running Expectation-Maximization')
 
        progress = PixelBar('Progress... ', max=corpus_size)

        # Collect counts
        for x in train_data:

            progress.next()

            tokens = tokenize( x.get_utterance() )
            labels = self.encode_to_iob( tokens, x.get_gold_slots() )


            # emission and unigram counts
            for w, s in zip( tokens, labels ):
                i = self.word2index[w]
                j = self.label2index[s]
                self.emit_p[i][j] += 1
                unigram_c[j] += 1

            # bigrams counts
            for bigram in self.get_bigrams( labels ):
                s_1 = self.label2index[bigram[0]]
                s = self.label2index[bigram[1]]
                bigram_c[s_1][s] += 1

            # trigram counts
            for trigram in self.get_trigrams( labels ):
                s_2 = self.label2index[trigram[0]]
                s_1 = self.label2index[trigram[1]]
                s = self.label2index[trigram[2]]
                self.trans_p[s_2][s_1][s] += 1

        print()
        print('Calculating probabilities from collected counts')
        # Normalize counts to get probabilities
        # Emission probabilities
        for w in range(n):
            for s in range(k-2):
                self.emit_p[w][s] = self.log_division( self.emit_p[w][s], unigram_c[s] )

        # Transition probabilities
        for s_2 in range(k-1):
            for s_1 in range(k-1):
                for s in range(k):
                    # trans_p[s_2][s_1][s] ==> P(s | s-2, s-1)
                    # so we need to normalize by count C(s-2, s-1)
                    self.trans_p[s_2][s_1][s] = self.log_division(self.trans_p[s_2][s_1][s], bigram_c[s_2][s_1])
        print('DONE')
        #print('WORDS: ', self.word2index.keys())
        #print('~~~~~~')
        #print('LABELS', self.label2index.keys())
        #print('~~~~~~')
        #print(self.emit_p)
        #print('~~~~~~')
        #print(self.trans_p)


    def log_division(self, a, b):
        return -50 if a==0 or b==0 else log2(a/b)

    def get_bigrams(self, sequence):
        """
        Params:
            sequence: list of strings
        Returns list of tuples
        """
        bigrams = []
        s_1 = '<START>'
        for s in sequence:
            bigrams.append( (s_1, s) )
            s_1 = s
        bigrams.append( (s_1, '<STOP>') )
        return bigrams


    def get_trigrams(self, sequence):
        """
        Params:
            sequence: list of strings
        Returns list of tripels (tuples of 3 elements)
        """
        trigrams = []
        s_1 = '<START>'
        s_2 = '<START>'
        for s in sequence:
            trigrams.append( (s_2, s_1, s) )
            s_2 = s_1
            s_1 = s
        trigrams.append( (s_2, s_1, '<STOP>') )
        return trigrams


    def save_model(self, fp):
        pass


    def test_model(self, test_data, corpus_size):

        print('Testing model...')
     
        self.correct = 0
        self.incorrect = 0
        self.total = 0

        progress = PixelBar('Progress... ', max=corpus_size)

        for x in test_data:

            progress.next()

            predictions = {}
            tokens = tokenize( x.get_utterance() )
            gold_labels = self.encode_to_iob( tokens, x.get_gold_slots() )
            
            """
            print('=' * 100)
            i += 1
            print('>>>', i)
            print( 'TEXT: ', x.get_utterance() )
            print( 'GOLD INTENT: ', x.get_gold_intent() )
            print( 'GOLD SLOTS: ', gold_labels)
            p = self.get_seq_probability(tokens, gold_labels)
            print( 'MODEL PROB: ', round( pow(2,p), 3) )
            print()
            """

            try:
                for intent in self.intent_set:
                    predictions[intent] = self.predict(x.get_utterance(), self.intent_set[intent], intent)
                #x.set_pred_slots_probs( predictions )


                self.get_scores( gold_labels, predictions[ x.get_gold_intent() ]['slots'] )
                
            except Exception:
                print('FAILED: [', x.get_gold_intent(), ']', x.get_utterance() )
            
        print()
        print('=' * 50)
        print('Total predicted slot labels: ', self.total)
        print('Accuracy: ', round(self.correct / self.total, 3) )
        print('=' * 50)

    def get_scores( self, gold_labels, pred_labels ):
        for g, p in zip( gold_labels, pred_labels ):
            if g == p:
                self.correct += 1
            else:
                self.incorrect += 1
            self.total += 1


    def predict(self, x, slots, intent):
        slots, prob = self.viterbi_search( tokenize(x), self.extract_iob_labels( slots ), intent )

        return {'slots': slots, 'prob': prob }


    def viterbi_search(self, x, states, intent):

        # Possible transition states at state i
        def S(i):
            return states if i>=0 else ['<START>']

        # Transition probability
        def Q(a, b, c):
            try:
                i, j, k = [self.label2index[s] for s in (a,b,c)]
                return self.trans_p[i][j][k]
            except Exception:
                return -50

        # Emission probability
        def E(w, s):
            try:
                w = self.word2index[w]
                s = self.label2index[s]
                return self.emit_p[w][s]
            except Exception:
                return -50

        # Path (highest sequence probability for each tag)
        P = {}
        # History of probabilities
        H = [{}]

        # Initialize with starting probabilities (t=0)
        t = 0
        for s in S(t):
            H[t][s] = {}
            H[t][s]['<START>'] = Q('<START>', '<START>', s) + E(x[t], s)
            P[s] = [s]

        # Run Viterbi for t > 0
        for t in range(1, len(x)):
            H.append({})
            newP = {}

            for s in S(t):
                # read s_1, s_2 as s-1, s-2
                H[t][s] = {}
                new_max = []
                for s_1 in S(t-1):
                    (max_p, max_s_2) = max( (H[t-1][s_1][s_2] + Q(s_2, s_1, s) + E(x[t], s), s_2) for s_2 in S(t-2) )
                    H[t][s][s_1] = max_p
                    new_max.append( (max_p, max_s_2, s_1) )

                _, _, max_s_1 = max(new_max)
                newP[s] = P[max_s_1] + [s]
            P = newP

        # Final step
        t = len(x)
        (max_p, max_s) = max( (H[t-1][s_1][s_2] + Q(s_2, s_1, '<STOP>'), s_1 ) for s_1 in S(t-1) for s_2 in S(t-2) )

        #print('X: ', x)
        #print('INTENT: ', intent, round( pow(2,max_p), 3) )
        #print('SLOT SEQUENCE: ', P[max_s])

        return P[max_s], max_p


    def get_seq_probability(self, sentence, labels):

        # Transition probability
        def Q(a, b, c):
            try:
                i, j, k = [self.label2index[s] for s in (a,b,c)]
                return self.trans_p[i][j][k]
            except Exception:
                return -50

        # Emission probability
        def E(w, s):
            try:
                w = self.word2index[w]
                s = self.label2index[s]
                return self.emit_p[w][s]
            except Exception:
                return -50

        p = 0
        s_2 = '<START>'
        s_1 = '<START>'
        for w, s in zip(sentence, labels):
            p += ( Q(s_2, s_1, s) + E(w, s) )
            s_2 = s_1
            s_1 = s

        # final prob
        p+= Q(s_2, s_1, '<STOP>')

        return p


    def extract_iob_labels(self, slot_labels):
        S = ['_O'] # Outside tag, i.e for tokens that are no entities
        for s in slot_labels:
            S.append(s+'_B') # Beginning tag
            S.append(s+'_I') # Inside tag
        return S


    def encode_to_iob(self, sentence, entities):

        """
        params:
            - sentence: list of tokens
            - entities: dict of slot => entity pairs
        """
        # base data structures: list of labels (target)
        iob_labels = ['_O' for i in range( len( sentence ) )]

        # first step: get base labels from slot entities
        for slot in entities:
            for token in tokenize( entities[slot] ):
                
                # TODO: if not in tokens?
                if token in sentence:
                    i = sentence.index(token)
                    iob_labels[i] = slot
                # DEBUG
                #else:
                #    print('MISSING: ', token)
                #    print(sentence)

        # redefine each label as either a beginning or inside tag
        
        # convert into list of tuples to make it easier to distinguish between single and sequence entities
        grouped_labels = []
        group = []
        prev_label = None
        for label in iob_labels:
            if label == prev_label:
                group.append(label)
            else:
                if prev_label:
                    grouped_labels.append(group)
                group = [label]
                prev_label = label
        if group:
            grouped_labels.append(group)

        # extract groups into one sequence
        labels = []
        for g in grouped_labels:
            if g[0] == '_O':
                labels += g
            elif len(g) == 1:
                labels.append( g[0] + '_I' )
            else:
                labels.append( g[0] + '_B' )
                labels += [x+'_I' for x in g[1:]]

        return labels


    def decode_from_iob(self, sentence, slots):
        # dict to store normalized entities
        entities = {}

        prev_label = None
        prev_sequence = []

        def flush_previous():
            nonlocal prev_label, prev_sequence
            if prev_label:
                entities[prev_label] = ' '.join(prev_sequence)
                prev_label = None
                prev_sequence = []
        
        for w, s in zip(sentence, slots):
            # Word is not entity, so no need to add
            if s.endswith('_O'):
                flush_previous()
                    
            # Word is beginning of sequence
            elif s.endswith('_B'):
                flush_previous()
                prev_label = s[:-2]
                prev_sequence.append(w)

            # Word is either single entity or continuation of previous
            elif s.endswith('_I'):
                # single entity
                if not prev_label:
                    entities[s[:-2]] = w
                # continuation
                else:
                    prev_sequence.append(w)
        flush_previous()

        return entities

    
if __name__ == '__main__':

    c = Corpus(20, 'train')
    c.get_data()
    intent_set, slot_set = corpus.get_labels()

    m = HMM_Slot_Filler()

    """
    # Test converting iob_labels to slot entities (dictionary)
    h = HMM_Slot_Filler()
    x = ['play', 'grand', 'requiem', 'by', 'mozart', 'now']
    s = ['_O', 'song_B', 'song_I', '_O', 'artist_I', 'time_I']
    e = h.decode_from_iob(x, s)
    print('input: ', x)
    print('labels: ', s)
    print('result: ', e)


    # Test encoding slot entities to iob_labels
    iob_labels = h.encode_to_iob(' '.join(x), e)
    print('\nconverting back to iob labels:')
    print('labels: ', iob_labels)
    #print('grouped: ', x)
    """