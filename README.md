# _Pythia-NLU_ : joint generative model for intent detection and slot filling

## Task Description

A dialog system usually has to perform two important tasks: understand the intent of an input sentence and identify the entities in the sentence that are important to response on this intent. These two tasks are respectively known as _intent detection_ and _slot filling_.

For example, the utterance __"Is there something new you can play by Lola Monroe?"__ has the intent `PlayMusic` and the slots `sort`, `artist` with the entities _"new"_, _"Lola Monroe"_ respectively.

## Dataset

We use the open-source [Snips NLU-benchmark](https://github.com/snipsco/nlu-benchmark) to train and test our model. The dataset contains seven intents with about 2000 instances for each, as well as a validation dataset with 100 instances for each intent.




| Intent                 | train data         | validation data	 |
|------------------------|--------------------|------------------|
| `SearchCreativeWork`   | 1,954              | 100              |
| `PlayMusic`            | 2,000              | 100              |
| `SearchScreeningEvent` | 1,959              | 100              |
| `GetWeather`           | 2,000              | 100              |
| `AddToPlaylist`        | 1,942              | 100              |
| `BookRestaurant`       | 1,973              | 100              |
| `RateBook`             | 1,956              | 100              |
| __Total__              | __13,784__         | __700__          |



## Model

Pythia-NLU is a generative model that takes advantage of the probabilities that intent parser and slot filler to input sentences. Mathematically we search for the intent that is assigned the highest probability by both sub-models and choose the subset of slots that are associated with this intent:

![chart](data/model_design.png)

where i is the predicted intent, __i__ is the set of possible intents, __e__ are the slot entities and __x__ is the sequence of tokens in the input sentence.

* [SVM](models/baseline_svm_intent.py) - Sklearn SVM as intent parser
* [HMM3](models/hmm3_slot_filler.py) - Home-made HMM with a trigram sequence model and viterbi search




## Experiments and Results

soon.
