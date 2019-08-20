
import random
class IntentParser:

    def __init__(self):
        pass

    def predict(self, corpus, intent_set):
        return [{i:random.uniform(0,1) for i in intent_set} for x in corpus]
