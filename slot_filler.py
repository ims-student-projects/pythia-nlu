
import random
class SlotFiller():

    def __init__(self):
        pass

    def predict(self, corpus, slot_set):

        predictions = []

        for x in corpus:
            prediction = {}

            for slot in slot_set:
                fake_probability = random.uniform(0,1)
                prediction[slot] = fake_probability
            predictions.append(prediction)

        return predictions
