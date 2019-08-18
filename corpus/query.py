class Query():
    def __init__(self, utterance, intent, slots):

        self.utterance = utterance
        self.gold_intent = intent
        self.pred_intent = None
        self.gold_slots = slots
        self.pred_slots = {}

    def get_gold_intent(self):
        return self.gold_intent

    def get_pred_intent(self):
        return self.pred_intent

    def set_pred_intent(self, __intent):
        self.pred_intent = __intent

    def get_gold_slots(self):
        return self.gold_slots

    def get_pred_slots(self):
        return self.pred_slots 

    def set_pred_slots(self, __slots):
        self.pred_slots = __slots
    
    def get_utterance(self):
        return self.utterance


    