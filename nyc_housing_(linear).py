
#dataset comes from https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market?resource=download

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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from scipy.stats import norm
from scipy.stats import boxcox
from scipy.stats.mstats import normaltest
from scipy import stats 


df = pd.read_csv(r'/Users/anuheaparker/Desktop/machine_learning/ml/NY-House-Dataset.csv')
#print(df)

#see first five of the df
#df.head(5)

#see info in the df 
#df.info()

#see more information of attribute
#print(df['PRICE'].describe())
#print(df["BATH"].value_counts())

#removed biggest outlier by price
print("this is the index of the highest price", df[['PRICE']].idxmax())
df = df.drop(labels=304,axis=0)

#CHECK FOR MISSING DATA AND DUPLICATES
#print(df.isnull().sum())
#print(sum(df.duplicated(subset = "ADDRESS")) == 0) #this dataset doesn't have unique ids
#print(df.index.is_unique)

#GRAPH OF COUNT OF A FEATURE 
#fig, ax = plt.subplots(figsize = (15,5))
#plt1 = sns.countplot(x=df['BEDS'], order=pd.value_counts(df['BEDS']).index)
#plt1.set(xlabel = 'Beds', ylabel='Count of Beds')
#plt.show()
#plt.tight_layout()

#SCATTERPLOT OF RELATIONSHIP WITH ALL FEATURES
#sns.pairplot(df)
#plt.show()

#see how many unique values each variable has 
df_uniques = pd.DataFrame([[i, len(df[i].unique())] for i in df.columns], columns=['Variable', 'Unique Values']).set_index('Variable')
print(df_uniques)

#separate out binary variables 
binary_variables = list(df_uniques[df_uniques['Unique Values'] == 2].index)
print(binary_variables)

#categorical variables
categorical_variables = list(df_uniques[(4582 >= df_uniques['Unique Values']) & (df_uniques['Unique Values'] > 2)].index)
print(categorical_variables)

[[i, list(df[i].unique())] for i in categorical_variables]

#if there are ordinal variables, pull them out 



#LINEAR ASSUMPTION TESTING 
#fig, (ax1, ax2) = plt.subplots(figsize = (12,8), ncols=2,sharey=False)
#sns.scatterplot( x = df.BEDS, y = df.PRICE,  ax=ax1)
#sns.regplot(x=df.BEDS, y=df.PRICE, ax=ax1)

#sns.scatterplot(x = df.BATH,y = df.PRICE, ax=ax2)
#sns.regplot(x=df.BATH, y=df.PRICE, ax=ax2);
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
    
#plotting_3_chart(df, 'PRICE')

previous_data = df.copy()

#LOG TRANSFORMATION & PLOTTING
df["PRICE"] = np.log(df["PRICE"])
#plotting_3_chart(df, "PRICE")

print(normaltest(df.PRICE.values))



cp_result = boxcox(previous_data.PRICE)
boxcox_price = cp_result[0]

print(normaltest(boxcox_price))

columns=['BROKERTITLE', 'TYPE', 'BEDS','BATH', 'PROPERTYSQFT','ADDRESS', 'STATE',
        'MAIN_ADDRESS', 'ADMINISTRATIVE_AREA_LEVEL_2', 'LOCALITY', 'SUBLOCALITY', 
        'STREET_NAME','LONG_NAME', 'FORMATTED_ADDRESS','LATITUDE','LONGITUDE', 'PRICE']

selected = df[columns]
selected.info()

categorical_columns=[key for key, value in selected.dtypes.items()  if value=='O']
#print(categorical_columns)

numeric_columns=list(set(columns)-set(categorical_columns))
#print(numeric_columns)

X = selected.drop("PRICE", axis=1)
X.head()

y = selected["PRICE"].copy()
y.head()

#for column in  categorical_columns:
#    print("column name:", column)
#    print("value_count:")
#    print( X[column].value_counts())

X_ = selected[categorical_columns+numeric_columns]

X_numeric=X_[numeric_columns].to_numpy() 

X_categorical=OneHotEncoder().fit_transform(X_[categorical_columns]).toarray() 

X_=np.concatenate((X_categorical,X_numeric), axis = 1)

df=pd.DataFrame(data=X_)
df.to_csv('cleaned_ny_housing_data.csv', index=False)


X_train, X_test, y_train, y_test = train_test_split( df, y, test_size=0.30, random_state=0)

ss = StandardScaler()

X_train = ss.fit_transform(X_train)

lm = LinearRegression()
lm.fit(X_train, y_train)

X_test=ss.transform(X_test)
price_predictions = lm.predict(X_test)
#print(price_predictions)

mse = mean_squared_error(y_test, price_predictions)
#print(mse)

lm.score(X_test,y_test)

print(r2_score(y_test, price_predictions))

