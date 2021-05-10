# Smm4hSharedTask
Source code for smm4h shared task 1 and task 2. Task 2 includes sub-tasks for English and French
## Dependency
* scikit-learn >= 0.22.1
* tensforflow == 2.0.0
* Keras == 2.3.1
* nltk == 3.4.5
* gensim == 3.8.3
## code description
* 10_fold_cross_val.py - 10-fold cross-validation training process for deep learning classifier based on Bi-LSTM architecture in task 1 and English sub-task for task 2
* 10_fold_cross_val_fr.py - 10-fold cross-validation training process for deep learning classifier based on Bi-LSTM architecture in task 1 and French sub-task for task 2
* BoWLR.py - the logistic regression classifier using features extracted by the Bag-of-Words (English task)
* BoWLR_fr.py - the logistic regression classifier using features extracted by the Bag-of-Words (French task)
* clean.py - pre-processing the corpus to train the word vector space models
* SentenceSimilarity.py - semantic similarity calculation of training data sets (by calculating the cosine value for mean word vector of the whole sentence in a tweet)
* data_select.py - select and down-sampling negative samples to select the most representative negative tweets (This script and the previous one are not very efficient)
* 
