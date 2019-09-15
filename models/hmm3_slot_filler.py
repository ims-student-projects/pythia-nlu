
"""
HMM3: Hidden Markov Model for seqence labelling with trigram model for label 
probabilities.

"""

import sys 
import numpy as np
from math import log
from nltk.tokenize import word_tokenize as tokenize

sys.path.append(sys.path[0] + '/../')
from progress.bar import PixelBar
from corpus.corpus_base import Corpus
from models.hmm3_viterbi import Viterbi


class HMM3():

    def __init__(self, slot_set, intent_set):
        """
        Initialize the Model, convert slot labels to IOB format.

        Args:
            slot_set:   list of all slots
            intent_set: dict, intent as key, list of slots as values
        """

        self.slot_set = slot_set
        self.intent_set = intent_set

        # Convert slots into list of IOB labels
        self.S = self.extract_iob_labels(slot_set)

        # Dicts that will map labels and tokens to index numbers
        self.label2index = {}
        self.word2index = {}

        # Transition probabilities (trigram probabilities for lables)
        self.emit_p = None
        # Emission probabilities (probability of a token given a label)
        self.trans_p = None


    def train_model(self, train_data):
        """
        Learns emission and transition probabilities from data

        Args:
            train_data:     iterable of instances, with the following method:
                get_size()

            Additionally each instance in train_data should have the methods:
                get_utterance()
                get_gold_slots()
        """

        print('Training HMM3 model')

        # Assign index numbers to all IOB labels
        self.label2index = { label:i for label, i in zip( self.S, range(len(self.S)) ) }
        self.label2index['<START>'] = len(self.label2index)
        self.label2index['<STOP>'] = len(self.label2index)

        # Assign index numbers to all tokens in corpus
        i = 0
        for x in train_data:
            for w in tokenize( x.get_utterance() ):
                if w not in self.word2index:
                    self.word2index[w] = i
                    i += 1

        print('Initializing probability matrixes')

        # Define probability and count matrixes
        k = len(self.label2index)
        n = len(self.word2index)
        self.emit_p = np.zeros( shape=(n, k-2) )
        self.trans_p = np.zeros( shape=(k-1, k-1, k) )

        unigram_c = np.zeros( shape=(k) )
        bigram_c = np.zeros( shape=(k-1, k) )

        print('Running Expectation-Maximization')
 
        progress = PixelBar('Progress... ', max=train_data.get_size())

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
        print('\nDONE')


    def log_division(self, a, b):
        """
        Returns the log division of two
        """
        return -50 if a==0 or b==0 else log(a/b)


    def get_bigrams(self, sequence):
        """
        Extract token bigrams
        Args:
            sequence: list of strings
        Returns:
            list of tuples of strings
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
        Extract token trigrams
        Args:
            sequence: list of strings
        Returns:
            list of tuples of strings
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


    def test_model(self, test_data):
        """
        Test the model on unseen data.

        Args:
            test_data:     iterable of instances, with the following method:
                get_size()

            Additionally each instance in train_data should have the methods:
                get_utterance()
        """

        print('Testing model...')
     
        correct_per_intent = 0
        total_per_intent = 0
        correct_all = 0
        total_all = 0
        self.viterbi = Viterbi(self.trans_p, self.emit_p, self.label2index, self.word2index)

        progress = PixelBar('Progress... ', max=test_data.get_size())

        for x in test_data:

            progress.next()

            tokens = tokenize( x.get_utterance() )
            gold_labels = self.encode_to_iob( tokens, x.get_gold_slots() )
 
            # Predict slots on intent level
            predictions = {}
            for intent in self.intent_set:
                predictions[intent] = self.predict(x.get_utterance(), self.intent_set[intent])
                x.set_pred_slots_per_intent(intent, self.decode_from_iob(tokens, predictions[intent]['slots']), predictions[intent]['prob'])
            c, t = self.get_scores( gold_labels, predictions[ x.get_gold_intent() ]['slots'] )
            correct_per_intent += c
            total_per_intent += t
                
            #except Exception as ex:
            #    print('\nFAILED: [', ex, ']', x.get_gold_intent(), ': ', x.get_utterance() )

            # Predict without considering intent
            prediction = self.predict(x.get_utterance(), self.slot_set)
            x.set_pred_slots( {'slots' : self.decode_from_iob(tokens, prediction['slots']), 'prob': prediction['prob'] } )
            c, t = self.get_scores( gold_labels, prediction['slots'] )            
            correct_all += c
            total_all += t

        print('\nDONE\n')
            
        accuracy_per_intent = (correct_per_intent / total_per_intent) if total_per_intent else 0
        print('=' * 50)
        print('Total predicted slots per intent: ', total_per_intent)
        print('Accuracy per intent: ', round(accuracy_per_intent, 3) )
        print('=' * 50)

        print()

        accuracy_all = ( correct_all / total_all ) if total_all else 0
        print('=' * 50)
        print('Total predicted slots: ', total_all)
        print('Accuracy: ', round(accuracy_all, 3) )
        print('=' * 50)


    def get_scores( self, gold_labels, pred_labels ):
        correct = 0
        total = 0
        for g, p in zip( gold_labels, pred_labels ):
            if g == p:
                correct += 1
            total += 1
        return correct, total


    def predict(self, x, labels):
        """
        Predict labels for a sequence

        Args:
            x:      string
            slots:  list of candidate labels

        Returns:
            Dict with two items:
            - slots: list of predicted labels
            - prob:  probability of the prediction

        """
        slots, prob = self.viterbi.search( tokenize(x), self.extract_iob_labels( labels ) )

        return {'slots': slots, 'prob': prob }


    def extract_iob_labels(self, slot_labels):        
        """
        Extract IOB-encoded labels

        Args:
            slots:      iterable of strings

        Returns:
            list of IOB labels

        """
        S = ['_O'] # Outside tag, i.e for tokens that are no entities
        for s in slot_labels:
            S.append(s+'_B') # Beginning tag
            S.append(s+'_I') # Inside tag
        return S


    def encode_to_iob(self, sentence, entities):

        """
        Extract IOB labels for a given sentence

        Args:
            - sentence: list of tokens
            - entities: dict of slot => entity items
        
        Returns:
            list of IOB labels
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
        """
        Convert IOB labels to entities

        Args:
            - sentence: list of tokens
            - slots:    list of IOB labels
        
        Returns:
            dict of slot => entity items
        """
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
