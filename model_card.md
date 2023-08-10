# Model Card


## Model Details
Mustafa Alturki  created the model. It is a Random Forest model that uses the default hyperparameters in scikit-learn 1.3.0

## Intended Use
This model should be used to predict whether a person has an income that is over $50k based on a number of attributes. The users of this model are those interested in income analysis based on demographic attributes.
## Training Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). 
The target class is a binary value indicating whether the individual's income is above or below $50,000.
The original dataset has 32,562 rows, and a 80-20 split was used to generate the train and test sets with no stratification. To use the data for training, categoriacl features were One-Hot encoded and labels were binarized.
## Evaluation Data
A 20% of the original dataset was used for evalution. The same Enconder and Binarizer used for training was applied to the evaluation set.
## Metrics
The model was evaluated using precision, recall and F-Beta Score. The values are as follows: Precision: 0.74, Recall: 0.64, F-Beta:0.69. 

## Ethical Considerations
Aequitas Bias Analysis shows the presence of bias at the unsupervised and supervised level, which indicates unfairness in the underlying data and the model trained using it.    
## Caveats and Recommendations
The modeling and training strategy was kept simple, since the focus of this project is the MLOps, specially deployment aspects. Therefore, the model predictions should not be considered for real-life scenarios that might affect individuals.