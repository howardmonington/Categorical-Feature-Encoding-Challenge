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

# Split up our columns between nominal, ordinal, binary, and time series
bin_cols = [col for col in X.columns.values if col.startswith('bin')]
num_cols = [col for col in X.columns.values if col.startswith('nom')]
ord_cols = [col for col in X.columns.values if col.startswith('ord')]
tim_cols = [col for col in X.columns.values if col.startswith('day') or col.startswith('month')]

# Checking to see how many unique values are in each column
X.nunique()

# Count of how many columns have each different type of dtype
X.dtypes.value_counts()

# Finding and plotting the count of the target variable
counts = y.value_counts()
plt.bar(counts.index, counts)
plt.gca().set_xticks([0,1])
plt.title('Distribution of Target Variable')
plt.show()
counts

# Creating a Logistic Regression algorithm with cross validation to be used to test the effectiveness of different techniques
def logistic(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LogisticRegression(solver='liblinear')
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    print('Accuracy : ' , accuracy_score(y_test, pred))
    
# One Hot Encoding
train_test_list = [X, test]
train_test_df = pd.concat(train_test_list)
train_test_df.shape

cat_cols = ['bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','ord_0','ord_1','ord_2','ord_3','ord_4','ord_5']

dummy_df = train_test_df[cat_cols]
train_test_df = train_test_df.drop(cat_cols, axis = 1)

# Get dummies
dummy_df = pd.get_dummies(dummy_df, sparse = False)

# Recombine dataframes
recombined_list = [train_test_df, dummy_df]
train_test_dummies = pd.concat(recombined_list, axis = 1)


# Split back into train and test matrices 
train = dummy_df.iloc[0:300000,:]
test = dummy_df.iloc[300000:,:]

# Cross Validation
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=test_size, random_state=seed)

X_train = scipy.sparse.csr_matrix(X_train)

# Using data to train AI
model = XGBClassifier()
model.fit(X_train, y_train)

# Testing accuracy of AI model
X_test = scipy.sparse.csr_matrix(X_test)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Using model to predict test data
test = scipy.sparse.csr_matrix(test)
pred = model.predict(test)

# Creating submission csv file
submission = pd.DataFrame(IDs, columns = ['id'])
submission['target'] = pred
submission.head()
submission.to_csv('submission_one_hot_encoding_v3.csv', index = False)
# pd.get_dummies

path = r'C:\Users\lukem\Desktop\Github AI Projects\Submissions\ '
submission.to_csv(path + 'updated_one_hot_encoding_v3.csv', index = False)
























































