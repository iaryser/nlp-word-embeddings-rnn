# Expected Model Performance

**Primary evaluation metric:** Test accuracy

### Model 1: Frozen Word Embeddings + 2-Layer Classifier

**Expected best test accuracy: 52%**

I expect this model to perform only slightly better than random guessing. Even though pretrained word embeddings can capture
semantic information, representing sentences through e.g. averaged embeddings loses word order and syntactic structure, which will
probably give the model a hard time of differentiating between plausible and implausible explanations. 

### Model 2: Frozen Word Embeddings + 2-Layer LSTM + Classifier

**Expected best test accuracy: 56%**

I expect the LSTM architecture to outperform the classifier-only model because it can process sequences of word embeddings and learn contextual representations that account for word order. However, abductive commonsense reasoning remains challenging, especially when training the LSTM from scratch with frozen embeddings and limited training data.

### Reference

Bhagavatula et al. (2020), *Abductive Commonsense Reasoning*. The original ART benchmark reports approximately 50% accuracy for simple embedding-based baselines and 58.8% for ESIM + ELMo. These results provide general context, although the course dataset is smaller and model architectures are different.