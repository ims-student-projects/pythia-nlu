
import random
class IntentParser:

    def __init__(self):
        pass

    def predict(self, corpus, intent_set):

        predictions = []

        for x in corpus:
            prediction = {}

            for intent in intent_set:
                fake_probability = random.uniform(0,1)
                prediction[intent] = fake_probability
            predictions.append(prediction)

        return predictions
