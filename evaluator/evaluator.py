
class Evaluator():

    """
    Calculates evaluation scores and renders string that can be printed on the console.
    Calculated scores:
        Model-level: F-macro, F-micro, Precision & Recall
        Intents and Slots: F, Precision & Recall

    Parameters:
        corpus: instance of Corpus
        intens: dict, where the keys are labels of instances
        slots:  list, with labels of slots
        epoch:  int (optional), epoch/iteration of training
    """

    def __init__(self, corpus, intents, slots, epoch=0):

        self.corpus = corpus
        self.intents = list(intents)
        self.slots = slots
        self.epoch = epoch

        # Print headers on console
        #self.print_headers(I, S, X)

        # Calculate TP, FP, FN, P, R, F1 for individual classes
        self.get_intent_scores()
        self.get_slot_scores()

        # Calculate F1 Macro and F1 Micro for model
        self.get_model_scores()


    def get_intent_scores(self):
        """
        Calculate scores for each individual intent:
            F, Precision, Recall
        """

        ## Initialize scores with 0s
        self.intent_tp = {i:0.0 for i in self.intents}
        self.intent_fp = {i:0.0 for i in self.intents}
        self.intent_fn = {i:0.0 for i in self.intents}
        self.intent_precision = {i:0.0 for i in self.intents}
        self.intent_recall = {i:0.0 for i in self.intents}
        self.intent_f1 = {i:0.0 for i in self.intents}

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
            p = self.intent_tp[i] / tp_fp if tp_fp else 0.0

            # recall = tp / (tp + fn)
            tp_fn = self.intent_tp[i] + self.intent_fn[i]
            r = self.intent_tp[i] / tp_fn if tp_fn else 0.0

            # f1 score = 2pr / p+r
            f1 = (2*p*r) / (p+r) if (p+r) else 0.0

            # Store calculated scores
            self.intent_precision[i] = p
            self.intent_recall[i] = r
            self.intent_f1[i] = f1


    def get_slot_scores(self):
        """
        Calculate scores for each individual Slot:
            F, Precision, Recall
        """
 

        ## Initialize base scores for slot values
        self.slot_tp = {s:0 for s in self.slots}
        self.slot_fp = {s:0 for s in self.slots}
        self.slot_fn = {s:0 for s in self.slots}
        self.slot_precision = {s:0.0 for s in self.slots}
        self.slot_recall = {s:0.0 for s in self.slots}
        self.slot_f1 = {s:0.0 for s in self.slots}

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
            p = self.slot_tp / tp_fp if tp_fp else 0.0

            # recall = tp / (tp + fn)
            tp_fn = self.slot_tp[s] + self.slot_fn[s]
            r = self.slot_tp[s] / tp_fn if tp_fn else 0.0

            # f1 score = 2pr / p+r
            f1 = (2*p*r) / (p+r) if (p+r) else 0.0

            # Store calculated scores
            self.slot_precision[s] = p
            self.slot_recall[s] = r
            self.slot_f1[s] = f1


    def get_model_scores(self):
        """
        Calculate Model-level scores:
            F-macro, F-micro, Precision, Recall
        """

        # F1-Macro is just average of all class F1 scores
        self.intent_f1['macro'] = sum( self.intent_f1.values() ) / len( self.intent_f1 )
        self.slot_f1['macro'] = sum( self.slot_f1.values() ) / len( self.slot_f1 )

        # F1-Micro is calculated from base scores of classes
        # Calculate for intent
        tp = sum( self.intent_tp.values() )
        fp = sum( self.intent_fp.values() )
        fn = sum( self.intent_fn.values() )
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (fp + fn) if (tp + fn) else 0.0
        self.intent_precision['micro'] = p
        self.intent_recall['micro'] = r
        self.intent_f1['micro'] = (2*p*r) / (p+r) if (p*r) else 0.0

        # Calculate for slots
        tp = sum( self.slot_tp.values() )
        fp = sum( self.slot_fp.values() )
        fn = sum( self.slot_fn.values() )
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (fp + fn) if (tp + fn) else 0.0
        self.slot_precision['micro'] = p
        self.slot_recall['micro'] = r
        self.slot_f1['micro'] = (2*p*r) / (p+r) if (p*r) else 0.0


    def resize(self, txt, length, direction='right'):
        """
        Adds whitespace to a string (either from left or right) to get
        the desired length. If string is shorter, just strip from right.
        """
        # convert to sting if necessary
        if type(txt) != str:
            txt = str(txt)

        # if string is larger than desired, strip
        if len(txt) > length:
            return txt[:length]
        # if shorter, add whitespace
        elif len(txt) < length:
            whitespace = ' ' *  (length - len(txt))
            return txt + whitespace if direction=='right' else whitespace + txt
        # string has already desired length
        else:
            return txt



    def __str__(self, use_bold=True):
        """
        Render string that can be printed into console.
        First line contains model-level scores. Second line contains scores of each individual 
        intent. Last two lines are scores for slots.
        """
        bold = '\033[1m' if use_bold else ''
        unbold = '\033[0m' if use_bold else ''

        epoch_header = '\nEpoch:   {} {}\n'.format(self.epoch, '#' * 179)
        # Model-level scores for Intent and Slot: F-macro, F-micro, precision and recall
        model_header = f'{bold}Intent:  Fmac   Fmic{unbold}   ' \
            f'Prc    Rec\t\t{bold}Slot: Fmac   Fmic{unbold}   Prc    Rec' + (' ' * 111)  + '#\n'

        model_scores = '         {}{}  {}{}  {}  {}\t\t      {}{}  {}{}  {}  {}{}#\n{}#\n'.format(
                        bold,
                        self.resize( round(self.intent_f1['macro'], 3), 5),
                        self.resize( round(self.intent_f1['micro'], 3), 5),
                        unbold,
                        self.resize( round(self.intent_precision['micro'], 3), 5),
                        self.resize( round(self.intent_recall['micro'], 3), 5),
                        bold,
                        self.resize( round(self.slot_f1['macro'], 3), 5),
                        self.resize( round(self.slot_f1['micro'], 3), 5),
                        unbold,
                        self.resize( round(self.slot_precision['micro'], 3), 5),
                        self.resize( round(self.slot_recall['micro'], 3), 5),
                        ' ' * 109,
                        ' ' * 189
                        )

        # Scores for each individual Intent: precision and recall
        intent_header = 'Intents: ' + '  '.join(self.resize(i,10) for i in self.intents) + (' ' * 98) + '#\n'
        intent_scores = '         ' + '  '.join(self.resize( '{}  {}'.format(
                        round(self.intent_precision[i], 2),
                        round(self.intent_recall[i], 2) ), 10 )
                for i in self.intents ) + (' ' * 98)  + '#\n' + (' ' * 189)  + '#\n'

        # Scores for each individual Slot: precision and recall
        # Split into to lines to avoid printing very long lines
        slots_header_a = 'Slots:   ' + ' '.join(self.resize(s,8) for s in self.slots[:20]) + ' #\n'
        slots_scores_a = '         ' + ' '.join(self.resize( '{} {}'.format(
                        round(self.slot_precision[s], 2),
                        round(self.slot_recall[s], 2) ), 8 )
                for s in self.slots[:20] ) + ' #\n' + (' ' * 189)  + '#\n'

        slots_header_b = 'Slots:   ' + ' '.join(self.resize(s,8) for s in self.slots[20:]) + (' ' * 10) + '#\n'
        slots_scores_b = '         ' + ' '.join(self.resize( '{} {}'.format(
                        round(self.slot_precision[s], 2),
                        round(self.slot_recall[s], 2) ), 8 )
                for s in self.slots[20:] ) +  (' ' * 10) + '#\n' + (' ' * 189)  + '#\n' + ('#' * 190)

        return ''.join([epoch_header, model_header, model_scores, intent_header, intent_scores, \
            slots_header_a, slots_scores_a, slots_header_b, slots_scores_b])