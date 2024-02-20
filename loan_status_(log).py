
#dataset comes from https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction

import tqdm as tqdm

import pandas as pd 
import numpy as np 

from itertools import accumulate

import matplotlib.pylab as plt 
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_digits, load_wine
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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


df = pd.read_csv(r'/Users/anuheaparker/Desktop/machine_learning/ml/loan_data.csv')
#print(df)

#see first five of the df
#print(df.head(5))

#see info in the df 
print(df.info())
#print(df.dtypes.value_counts())

#see more information of attribute
#print(df['Loan_Status'].describe())
#print(df["LoanAmount"].value_counts())

#CHECK FOR MISSING DATA AND DUPLICATES
#print(df.isnull().sum()) #gender, self_employed, loan_amount_term, credit_history have missing data
df = df.dropna(axis=0)
#print(sum(df.duplicated(subset = "Loan_ID")) == 0) #no duplicates in this dataset
#print(df.index.is_unique)

#GRAPH OF COUNT OF A FEATURE 
#fig, ax = plt.subplots(figsize = (15,5))
#plt1 = sns.countplot(x=df['Gender'], order=pd.value_counts(df['Gender']).index)
#plt1.set(xlabel = 'Gender', ylabel='Count of Gender')
#plt.show()
#plt.tight_layout()

#SCATTERPLOT OF RELATIONSHIP WITH ALL FEATURES
#sns.pairplot(df)
#plt.show()

#SKEW VARIABLES

# Create a list of float colums to check for skewing
mask = df.dtypes == float
float_cols = df.columns[mask]

skew_limit = 0.75 # define a limit above which we will log transform
skew_vals = df[float_cols].skew()

# Showing the skewed columns
skew_cols = (skew_vals
            .sort_values(ascending=False)
            .to_frame()
            .rename(columns={0:'Skew'})
            .query('abs(Skew) > {}'.format(skew_limit)))

#print("these are the skew cols", skew_cols) - I'm not transforming this because it sucks and nothing is working for the left skewed ones 


data_copy = df

#CHANGE ID TO NUMBER (if need to use this for later analysis)
df["Loan_ID"] = df["Loan_ID"].str.replace('LP','')
df["Loan_ID"] = df["Loan_ID"].astype(int)

df = df.drop("Loan_ID", axis=1)

#ONE HOT ENCODING 
df = pd.get_dummies(data=df, columns = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"])
print(df.head(5))

df['Loan_Status_Encoded'], class_names = pd.factorize(df['Loan_Status'])

df.drop(columns=['Loan_Status'], inplace=True)

print(df.head(5))

#see how many unique values each variable has 
#df_uniques = pd.DataFrame([[i, len(df[i].unique())] for i in df.columns], columns=['Variable', 'Unique Values']).set_index('Variable')
#print(df_uniques)

#separate out binary variables 
#binary_variables = list(df_uniques[df_uniques['Unique Values'] == 2].index)
#print(binary_variables)

#categorical variables
#categorical_variables = list(df_uniques[(4582 >= df_uniques['Unique Values']) & (df_uniques['Unique Values'] > 2)].index)
#print(categorical_variables)

#[[i, list(df[i].unique())] for i in categorical_variables]

#if there are ordinal variables, pull them out 

#LINEAR ASSUMPTION TESTING 
#fig, (ax1, ax2) = plt.subplots(figsize = (12,8), ncols=2,sharey=False)
#sns.scatterplot( x = df.LoanAmount, y = df.Loan_Status_Encoded,  ax=ax1)
#sns.regplot(x=df.LoanAmount, y=df.Loan_Status_Encoded, ax=ax1)
#plt.show()

#sns.scatterplot(x = df.Education,y = df.Loan_Status, ax=ax2)
#sns.regplot(x=df.Education, y=df.Loan_Status, ax=ax2);
#plt.show()


#CHECK IF TARGET IS NORMALLY DISTRIBUTED 
def plotting_3_chart(data, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(data.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(data.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(data.loc[:,feature], orient='v', ax = ax3);
    
    plt.show()
    
#plotting_3_chart(df, 'Loan_Status_Encoded')

#previous_data = df.copy()

#LOG TRANSFORMATION & PLOTTING
#df["Loan_Status_Encoded"] = np.log(df["Loan_Status_Encoded"])
#plotting_3_chart(df, "Loan_Status_Encoded")

#print(normaltest(df.Loan_Status_Encoded.values))





#MODEL TESTING


X = df.drop("Loan_Status_Encoded", axis=1)
y = df["Loan_Status_Encoded"]


corr_matrix = X.corr()
print(corr_matrix)





scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca=PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.3, random_state=42)

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

