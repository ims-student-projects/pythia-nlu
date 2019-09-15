class Query():
    def __init__(self, utterance, intent, slots):

        self.utterance = utterance
        self.gold_intent = intent
        self.gold_slots = slots
        self.pred_intent = None
        self.intent_probabilities = {}
        self.pred_slots = {}
        self.pred_slots_per_intent = {}

    def get_utterance(self):
        return self.utterance

    def get_gold_intent(self):
        return self.gold_intent

    def get_gold_slots(self):
        return self.gold_slots

    def set_pred_intent(self, intent):
        self.pred_intent = intent

    def get_pred_intent(self):
        return self.pred_intent

    def set_intent_probabilities(self, probs):
        self.intent_probabilities = probs

    def get_intent_probabilities(self):
        return self.intent_probabilities


    def set_pred_slots(self, slots):
        self.pred_slots = slots
    
    def get_pred_slots(self):
        return self.pred_slots 

    def set_pred_slots_per_intent(self, intent, slots, prob):
        self.pred_slots_per_intent[intent] = {}
        self.pred_slots_per_intent[intent]['slots'] = slots
        self.pred_slots_per_intent[intent]['prob'] = prob

    def get_pred_slots_per_intent(self):
        return self.pred_slots_per_intent

    def __str__(self):
        return f'[{self.gold_intent}]\t"{self.utterance}"\t({self.gold_slots})'
    