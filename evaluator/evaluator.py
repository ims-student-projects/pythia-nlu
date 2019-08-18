# ------ baseline evaluator ------ #

class Evaluator():

    def __init__(self, corpus, intents, slots):

        self.corpus = corpus
        self.intents = intents
        self.slots = slots

        # Print headers on console
        #self.print_headers(I, S, X)

        # Calculate TP, FP, FN, P, R, F1 for individual classes
        self.get_intent_scores()
        self.get_slot_scores()

        # Calculate F1 Macro and F1 Micro for model
        self.get_model_scores()


    def get_intent_scores(self):

        ## Initialize scores with 0s
        self.intent_tp = {i:0 for i in self.intents}
        self.intent_fp = {i:0 for i in self.intents}
        self.intent_fn = {i:0 for i in self.intents}
        self.intent_precision = {i:0 for i in self.intents}
        self.intent_recall = {i:0 for i in self.intents}
        self.intent_f1 = {i:0 for i in self.intents}

        # Calculate base scores: TP, FP and FN
        for x in self.corpus:
            for i in self.intents:
                if x.get_pred_intent() == i:
                    # predicted intent is true
                    if x.get_gold_intent() == i:
                        self.intent_tp[i] += 1
                    # predicted intent is false
                    else:
                        self.intent_fp[i] += 1
                # intent is true but not predicted
                elif x.get_gold_intent() == i:
                    self.intent_fn[i] += 1

        # Calculate precision, recall, local F score
        for i in self.intents:
            # precision = tp / (tp + fp)
            tp_fp = self.intent_tp[i] + self.intent_fp[i]
            p = self.intent_tp[i] / tp_fp if tp_fp else 0

            # recall = tp / (tp + fn)
            tp_fn = self.intent_tp[i] + self.intent_fn[i]
            r = self.intent_tp[i] / tp_fn if tp_fn else 0

            # f1 score = 2pr / p+r
            f1 = (2*p*r) / (p+r) if (p+r) else 0

            # Store calculated scores
            self.intent_precision[i] = p
            self.intent_recall[i] = r
            self.intent_f1[i] = f1


    def get_slot_scores(self):

        ## Initialize base scores for slot values
        self.slot_tp = {s:0 for s in self.slots}
        self.slot_fp = {s:0 for s in self.slots}
        self.slot_fn = {s:0 for s in self.slots}
        self.slot_precision = {s:0 for s in self.slots}
        self.slot_recall = {s:0 for s in self.slots}
        self.slot_f1 = {s:0 for s in self.slots}

        # Calculate base scores: TP, FP and FN
        for x in self.corpus:
            gold = x.get_gold_slots()
            pred = x.get_pred_slots()
            # Iterate gold slots
            for s in gold:
                if s in pred and pred[s]:
                    if gold[s] == pred[s]:
                        self.slot_tp[s] += 1
                    else:
                        self.slot_fp[s] += 1
                else:
                    self.slot_fn[s] +=1
            for s in pred:
                if s not in gold:
                    self.slot_fp[s] += 1

        # Calculate precision, recall, local F score
        for s in self.slots:
            # precision = tp / (tp + fp)
            tp_fp = self.slot_tp[s] + self.slot_fp[s]
            p = self.slot_tp / tp_fp if tp_fp else 0

            # recall = tp / (tp + fn)
            tp_fn = self.slot_tp[s] + self.slot_fn[s]
            r = self.slot_tp[s] / tp_fn if tp_fn else 0

            # f1 score = 2pr / p+r
            f1 = (2*p*r) / (p+r) if (p+r) else 0

            # Store calculated scores
            self.slot_precision[s] = p
            self.slot_recall[s] = r
            self.slot_f1[s] = f1


    def get_model_scores(self):

        # F1-Macro is just average of all class F1 scores
        self.intent_f1['macro'] = sum( self.intent_f1.values() ) / len( self.intent_f1 )
        self.slot_f1['macro'] = sum( self.slot_f1.values() ) / len( self.slot_f1 )

        # F1-Micro is calculated from base scores of classes
        # Calculate for intent
        tp = sum( self.intent_tp.values() )
        fp = sum( self.intent_fp.values() )
        fn = sum( self.intent_fn.values() )
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (fp + fn) if (tp + fn) else 0
        self.intent_f1['micro'] = (1*p*r) / (p*r) if (p*r) else 0

        # Calculate for slots
        tp = sum( self.slot_tp.values() )
        fp = sum( self.slot_fp.values() )
        fn = sum( self.slot_fn.values() )
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (fp + fn) if (tp + fn) else 0
        self.slot_f1['micro'] = (1*p*r) / (p*r) if (p*r) else 0


    def __str__(self, use_bold=True):
        """
        Fancy list of macro, micro f-scores and precision and recall for all labels
        """
        bold = '\033[1m' if use_bold else ''
        unbold = '\033[0m' if use_bold else ''

        # Macro and Micro F1 scores of intent prediction
        intent_sum = '{}{}  {}{}\t'.format(
                        bold,
                        round(self.intent_f1['macro'], 2),
                        round(self.intent_f1['micro'], 2),
                        unbold  )

        # F1, Precision and Recall for each intent
        intent_scores = '\t'.join('{} {} {}'.format(
                        round(self.intent_f1[i], 2),
                        round(self.intent_precision[i], 2),
                        round(self.intent_recall[i], 2) )
                for i in self.intents )

        # Macro and Micro F1 scores of slot filling
        slot_sum = '{}{}  {}{}\t'.format(
                        bold,
                        round(self.slot_f1['macro'], 2),
                        round(self.slot_f1['micro'], 2),
                        unbold  )

        # F1, Precision and Recall for each intent
        slot_scores = '\t'.join('{} {} {}'.format(
                        round(self.slot_f1[s], 2),
                        round(self.slot_precision[s], 2),
                        round(self.slot_recall[s], 2) )
                for s in self.slots )

        return ''.join([intent_sum, intent_scores, slot_sum, slot_scores])
