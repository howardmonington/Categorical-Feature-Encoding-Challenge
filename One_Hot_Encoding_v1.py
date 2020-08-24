import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
pd.set_option('max_columns', None)
import xgboost
from xgboost import XGBClassifier

# Get train and test dataset
df_train = pd.read_csv('C:/Users/lukem/Desktop/AI Projects/Categorical Feature Encoding Challenge/train.csv')
df_test = pd.read_csv('C:/Users/lukem/Desktop/AI Projects/Categorical Feature Encoding Challenge/test.csv')

# Set up our X and y for our training set
X = df_train.drop(columns=['id', 'target'])
y = df_train['target']
test = df_test.drop(columns=['id'])
labels = X.columns
IDs = df_test['id']

print("Training set shape: {} \nTest set shape: {}".format(X.shape, test.shape))



# Creating submission csv file
submission = pd.DataFrame(IDs, columns = ['id'])
submission['target'] = pred
submission.head()
submission.to_csv('submission_one_hot_encoding_v2.csv', index = False)
# pd.get_dummies

























































