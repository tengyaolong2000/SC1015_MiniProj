# Welcome to Hypertension Analysis Repository!
## SC1015 mini project 

# IMPORTANT!!!
Paste notebook in https://nbviewer.org/ to see full notebook. Some charts only work there and not shown on GitHub since GitHub only shows static images.
Alternatively, just click [THIS LINK HERE](https://nbviewer.org/github/tengyaolong2000/SC1015_MiniProj/blob/main/hypert_pred.ipynb)
## About
This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence). We use the [Stroke Dataset from Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

The order of our notebook is:
1) Introduction
2) Exploratary Data Analysis
3) Data Balancing
4) Modeling
5) Final Thoughts

(Links to jump around the notebook will be provided in the notebook itself to minimise scrolling. However it only works on https://nbviewer.org/ !)

## Contributors
- @tengyaolong2000 Teng Yao Long
- @jewel-chin Jewel Chin
- @yuminp Park Yumin

## Problem Definition
- What are the main predictors of Hypertension?
- Which model would be the best to predict Hypertension?

## Models Used
1) Logistic Regression
2) Decision Tree
3) Random Forest
4) Support Vector Machine
5) Artificial Neural Network
6) eXtreme Gradient Boosting Classifier
7) K Nearest Neighbours 
8) Naive Bayes Classifier

## Conclusion
- Age and BMI unanimously are the biggest predictors of hypertension
- Other predictors include average glucose level and heart disease (It's important to exercise!!! 🏃‍♂️🏃‍♀️)
- Tree models are good are predicting hypertension if we focus only on recall. However the scores of other metrics are sacrificed too much.
- Logistic Regression and Naive Bayes models have decent recall without sacrificing other scores too much. These models are good if we have limited resources (GPU/ memory)
- If we have sufficient resources, the Neural Network has the potential to be the best after more hyperparameter tuning/ increase in model complexity
- However we also need to deal with overfitting.
- If we were to use a Deep Learning approach, we could also utilise transfer learning/ ensemble modeling.

## What did we learn from this project?
1) Handling imbalanced datasets using resampling methods and imblearn package (SMOTE)
2) Feature selection/ feature importance techniques (RFE, SHAP, Permutation importance)
3) Logistic Regression with sklearn
4) Random Forest with sklearn
5) Support Vector Machines with sklearn
6) Aritficial Neural Networks with TensorFlow Keras
7) XGBoost with xgboost
8) K Nearest Neighbours with sklearn
9) Naive Bayes Classifier with sklearn
10) Collaborating using GitHub
11) Data visualisation with plotly
12) Grid Search to determine best hyperparameters
13) Concepts on different metrics such as Recall, F1 score

## References
- https://en.wikipedia.org/wiki/Artificial_neural_network
- https://en.wikipedia.org/wiki/Naive_Bayes_classifier
- http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://keras.io/
- https://plotly.com/python/
- https://en.wikipedia.org/wiki/Logistic_regression
- https://en.wikipedia.org/wiki/Random_forest
- https://scikit-learn.org/stable/modules/svm.html
- https://en.wikipedia.org/wiki/Support-vector_machine
- https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
- https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
- https://en.wikipedia.org/wiki/Naive_Bayes_classifier
- https://scikit-learn.org/stable/modules/naive_bayes.html
- https://en.wikipedia.org/wiki/XGBoost
- https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
- https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/
- https://machinelearningmastery.com/rfe-feature-selection-in-python/
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
- https://shap.readthedocs.io/en/latest/index.html
- https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
- https://scikit-learn.org/stable/modules/permutation_importance.html#:~:text=The%20permutation%20feature%20importance%20is,model%20depends%20on%20the%20feature.


