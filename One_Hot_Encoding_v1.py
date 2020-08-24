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
from imblearn.over_sampling import RandomOverSampler

# Get train and test dataset
df_train = pd.read_csv('C:/Users/lukem/Desktop/AI Projects/Categorical Feature Encoding Challenge/train.csv')
df_test = pd.read_csv('C:/Users/lukem/Desktop/AI Projects/Categorical Feature Encoding Challenge/test.csv')
sample_submission = pd.read_csv(r'C:\Users\lukem\Desktop\AI Projects\Categorical Feature Encoding Challenge\sample_submission.csv')

sample_submission.head()
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
train_test_df = pd.concat(train_test_list, axis = 0)
train_test_df.shape

train_test_df[['bin_3','bin_4']] = train_test_df[['bin_3','bin_4']].replace({'Y':1,'N':0,'T':1,'F':0})

ord_1_dictionary = {'Novice':0,'Contributor':1,'Expert':2,'Master':3,'Grandmaster':4}
train_test_df['ord_1'] = train_test_df['ord_1'].map(ord_1_dictionary)

ord_2_dictionary = {'Freezing':0,'Cold':1,'Warm':2,'Hot':3,'Boiling Hot':4,'Lava Hot':5}
train_test_df['ord_2'] = train_test_df['ord_2'].map(ord_2_dictionary)

train_test_df.ord_3.unique()
ord_3_dictionary = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14}
train_test_df['ord_3'] =  train_test_df['ord_3'].map(ord_3_dictionary)

train_test_df.ord_4.unique()
ord_4_dictionary = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25}
train_test_df['ord_4'] = train_test_df['ord_4'].map(ord_4_dictionary)

train_test_df.ord_5.unique()
ord_5_dictionary = {'ac':1,'av':2,'be':3,'ck':4,'cp':5,'dh':6,'eb':7,'eg':8,'ek':9,'ex':10,'fh':11,'hh':12,'hp':13,'ih':14,'je':15,
'jp':16,'ke':17,'kr':18,'kw':19,'ll':20,'lx':21,'mb':22,'mc':23,'mm':24,'nh':25,'od':26,'on':27,'pa':28,'ps':29,'qo':30,
'qv':31,'qw':32,'ri':33,'rp':34,'sn':35,'su':36,'tv':37,'ud':38,'us':39,'ut':40,'ux':41,'uy':42,'vq':43,'vy':44,
'wu':45,'wy':46,'xy':47,'yc':48,'aF':49,'aM':50,'aO':51,'aP':52,'bF':53,'bJ':54,'cA':55,'cG':56,'cW':57,'dB':58,
'dE':59,'dN':60,'dO':61,'dP':62,'dQ':63,'dZ':64,'eG':65,'eQ':66,'fO':67,'gJ':68,'gM':69,'hL':70,'hT':71,'iT':72,
'jS':73,'jV':74,'kC':75,'kE':76,'kK':77,'kL':78,'kU':79,'kW':80,'lF':81,'lL':82,'nX':83,'oC':84,'oG':85,'oH':86,
'oK':87,'qA':88,'qJ':89,'qK':90,'qP':91,'qX':92,'rZ':93,'sD':94,'sV':95,'sY':96,'tM':97,'tP':98,'uJ':99,'uS':100,
'vK':101,'xP':102,'yN':103,'yY':104,'zU':105,'Ai':106,'Aj':107,'Bb':108,'Bd':109,'Bn':110,'Cl':111,'Dc':112,'Dx':113,'Ed':114,
'Eg':115,'Er':116,'Fd':117,'Fo':118,'Gb':119,'Gx':120,'Hj':121,'Id':122,'Jc':123,'Jf':124,'Jt':125,'Kf':126,'Kq':127,'Mf':128,
'Ml':129,'Mx':130,'Nf':131,'Nk':132,'Ob':133,'Os':134,'Ps':135,'Qb':136,'Qh':137,'Qo':138,'Rm':139,'Ry':140,'Sc':141,'To':142,
'Uk':143,'Uu':144,'Vf':145,'Vx':146,'Wc':147,'Wv':148,'Xh':149,'Xi':150,'Yb':151,'Ye':152,'Zc':153,'Zq':154,'AP':155,'BA':156,
'BE':157,'CL':158,'CM':159,'CU':160,'CZ':161,'DH':162,'DN':163,'FI':164,'GD':165,'GJ':166,'IK':167,'JX':168,'KR':169,'KZ':170,
'LE':171,'MC':172,'MO':173,'MV':174,'NV':175,'OR':176,'PA':177,'PQ':178,'PZ':179,'QM':180,'RG':181,'RL':182,'RP':183,'SB':184,
'TR':185,'TZ':186,'UO':187,'WE':188,'XI':189,'YC':190,'ZR':191,'ZS':192}
train_test_df['ord_5'] = train_test_df['ord_5'].map(ord_5_dictionary)

cat_cols = ['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','ord_3','ord_4','ord_5']

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

# oversample
oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X_train, y_train)

X_train = scipy.sparse.csr_matrix(X_train)

# Using data to train AI
model = XGBClassifier(
    learning_rate = 0.1,
    n_estimators = 500,
    max_depth = 7,
    min_child_weight = 0.5)

model.fit(X_train, y_train)
X_test = scipy.sparse.csr_matrix(X_test)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


xgb2 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
xgb2.fit(X_train, y_train)

# Testing accuracy of AI model
X_test = scipy.sparse.csr_matrix(X_test)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Using model to predict test data
test = scipy.sparse.csr_matrix(test)


# Creating submission csv file
submission = pd.DataFrame(IDs, columns = ['id'])
submission['target'] = pred
submission.head()
submission.to_csv('submission_one_hot_encoding_v3.csv', index = False)
# pd.get_dummies

path = r'C:\Users\lukem\Desktop\Github AI Projects\Submissions\ '
submission.to_csv(path + 'xgb_submission_v6.csv', index = False)
























































