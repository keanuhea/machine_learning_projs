
#dataset comes from https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction

import tqdm as tqdm
import pandas as pd 
from pandas.plotting import table
import numpy as np 
import matplotlib.pylab as plt 
import matplotlib.pyplot as plot
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits, load_wine
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from scipy.stats import norm
from scipy.stats import boxcox
from scipy.stats import yeojohnson
from scipy.stats.mstats import normaltest
from scipy import stats 




df = pd.read_csv(r'/Users/anuheaparker/Desktop/ml/loan_data.csv')

pd.set_option("display.max.columns", None)

#see df info
print(df.head(5))

#don't need the unique loan ID in this dataset 
df = df.drop(['Loan_ID'], axis=1)

#look at more info
print(df.info())
print(df.dtypes.value_counts())

#check for missing data and duplicates 
print(df.isnull().sum()) 
#gender, self_employed, loan_amount_term, credit_history have missing data
#print(sum(df.duplicated(subset = "Loan_ID")) == 0) #no duplicates in this dataset

#originally just tried deleting the rows with missing data, but realized that would drop too much
#replace missing data values with mode 

df["Gender"] = df["Gender"].fillna(df["Gender"].mode().iloc[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode().iloc[0])
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode().iloc[0])
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode().iloc[0])

#dependents need to be changed to a numeric value
df["Dependents"] = df["Dependents"].replace(['0', '1', '2', '3+'], [0,1,2,3])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode().iloc[0])

df["CoapplicantIncome"] = df["CoapplicantIncome"].astype(int)
df["LoanAmount"] = df["LoanAmount"].astype(int)

print(df.isnull().sum())



def make_bar_chart(feature):
    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    sns.countplot(x=df[feature], palette="flare")
    plt.subplot(2,2,1)
    plt.show()

make_bar_chart("Gender")
make_bar_chart("Dependents")
make_bar_chart("Self_Employed")




#one hot encoding
df = pd.get_dummies(data=df, columns = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"])
print(df.head(5))

df['Loan_Status_Encoded'], class_names = pd.factorize(df['Loan_Status'])

df.drop(columns=['Loan_Status'], inplace=True)



numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

fig,axes = plt.subplots(1,3,figsize=(17,5))
for idx,cat_col in enumerate(numerical_columns):
    sns.boxplot(y=cat_col,data=df,x='Loan_Status_Encoded',ax=axes[idx], palette="flare")

print(df[numerical_columns].describe())
plt.subplots_adjust(hspace=1)
plt.show()

plt.figure(figsize=(6, 6))
    
corr = df.select_dtypes(include=np.number).corr()
mask = np.triu(np.ones_like(corr))
    
sns.heatmap(corr, vmin=-1, vmax=1, mask=mask, cmap='flare', annot=True, fmt='.2f', linewidths=0.1, annot_kws={'weight':'bold'})
    
plt.title('Correlation Heatmap', fontdict={'fontsize':'x-large', 'fontweight':'bold'})
plt.show()




#MODEL TESTING

X = df.drop("Loan_Status_Encoded", axis=1)
y = df["Loan_Status_Encoded"]


corr_matrix = X.corr()
print(corr_matrix)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

#scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Fit Logistic Regression model without regularization
log_reg = LogisticRegression(penalty='none', solver='saga', random_state=42)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
print("Logistic Regression (No Regularization) Accuracy:", log_reg_accuracy)

# Fit Lasso Logistic Regression model (L1 regularization)
lasso_log_reg = LogisticRegression(penalty='l1', solver='saga', random_state=42)
lasso_log_reg.fit(X_train, y_train)
lasso_log_reg_pred = lasso_log_reg.predict(X_test)
lasso_log_reg_accuracy = accuracy_score(y_test, lasso_log_reg_pred)
print("Lasso Logistic Regression Accuracy:", lasso_log_reg_accuracy)

# Fit Ridge Logistic Regression model (L2 regularization)
ridge_log_reg = LogisticRegression(penalty='l2', solver='saga', random_state=42)
ridge_log_reg.fit(X_train, y_train)
ridge_log_reg_pred = ridge_log_reg.predict(X_test)
ridge_log_reg_accuracy = accuracy_score(y_test, ridge_log_reg_pred)
print("Ridge Logistic Regression Accuracy:", ridge_log_reg_accuracy)

# Fit Elastic Net Logistic Regression model (L1 and L2 regularization)
elastic_net_log_reg = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42)
elastic_net_log_reg.fit(X_train, y_train)
elastic_net_log_reg_pred = elastic_net_log_reg.predict(X_test)
elastic_net_log_reg_accuracy = accuracy_score(y_test, elastic_net_log_reg_pred)
print("Elastic Net Logistic Regression Accuracy:", elastic_net_log_reg_accuracy)

print("precision", precision_score(y_test,log_reg_pred))
print("recall", recall_score(y_test,log_reg_pred))
print("f1", f1_score(y_test,log_reg_pred))
print("auc", roc_auc_score(y_test,log_reg_pred))

#random forest classifier 
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

#gradient boosting classifier 
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", accuracy_xgb)

#svm classifier 
svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

#neural networks multi-layer perceptron 
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000, random_state=42)
mlp_classifier.fit(X_train, y_train)
y_pred_mlp = mlp_classifier.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("MLP Accuracy:", accuracy_mlp)

#k nearest neighbors 
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)

#ensemble methods 
voting_classifier = VotingClassifier(estimators=[('rf', random_forest), ('xgb', xgb_classifier), ('svm', svm_classifier)], voting='hard')
voting_classifier.fit(X_train, y_train)
y_pred_voting = voting_classifier.predict(X_test)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print("Voting Classifier Accuracy:", accuracy_voting)

#decision tree classifier 
dtree = DecisionTreeClassifier(max_depth=3, min_samples_leaf= 35)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
accuracy_dtree = accuracy_score(y_test, y_pred_dtree)
roc_score_dtree = roc_auc_score(y_test, y_pred_dtree)
print("Dtree accuracy: ", accuracy_dtree)
