# Import required libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# Set random seed
np.random.seed(42)

# Load csv file
df = pd.read_csv('framingham.csv')


# view top 5 rows
df.head()

# Visualize male/female ration 
sns.countplot(x=df["male"]).set_title("Male/Female Ratio")

# Visualize the classes distributions 
sns.countplot(x=df["TenYearCHD"]).set_title("Outcome Count")

# Visualize the classes distributions by gender 
sns.countplot(x="TenYearCHD", hue="male", data=df).set_title('Outcome Count by Gender')

# Check if there any null values 
df.isnull().values.any()

# Remove null values 
df = df.dropna()

# Check if there any null values
df.isnull().values.any()

# Specify features columns
X = df.drop(columns="TenYearCHD", axis=0)

# Specify target column 
y = df["TenYearCHD"]

# Import required library for resampling 
from imblearn.under_sampling import RandomUnderSampler

# Instantiate Random Under Sampler
rus = RandomUnderSampler(random_state=42)

# Perform random under sampling 
df_data, df_target = rus.fit_resample(X,y)

# Visualize new classes distributions 
sns.countplot(df_target).set_title('Balanced Data Set')

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from IPython.display import display

# Define dictionary with performance metrics 
scoring = {'accuracy':make_scorer(accuracy_score),
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score),
           'f1_score':make_scorer(f1_score)}
        
# Import required libraries for machine learning classifiers

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Instantiate the machine learning classifiers
log_model = LogisticRegression(max_iter=10000)
svc_model = LinearSVC(dual=False)
dtr_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()
gnb_model = GaussianNB()

# Dfine the models evaluation function 
def models_evaluation(X, y, folds):
    
    '''
    X = data set features
    y : data set target 
    folds : number of cross-validation folds
    '''
    
    # Perform cross-validatio to each machine learning classifier
    
    log = cross_validate(log_model,X, y, cv=folds, scoring=scoring)
    svc = cross_validate(log_model,X, y, cv=folds, scoring=scoring)
    dtr = cross_validate(log_model,X, y, cv=folds, scoring=scoring)
    rfc = cross_validate(log_model,X, y, cv=folds, scoring=scoring)
    gnb = cross_validate(log_model,X, y, cv=folds, scoring=scoring)
    
    # Create a data frame with the models performance scores
    models_scores_table = pd.DataFrame({'Logistic Regression':[log['test_accuracy'].mean(),
                                                               log['test_precision'].mean(),
                                                               log['test_recall'].mean(),
                                                               log['test_f1_score'].mean()],
                                       
                                        'Support Vector Classification':[svc['test_accuracy'].mean(),
                                                                         svc['test_precision'].mean(),
                                                                         svc['test_recall'].mean(),
                                                                         svc['test_f1_score'].mean()],
                                        
                                        'Decision Tree':[dtr['test_accuracy'].mean(),
                                                         dtr['test_precision'].mean(),
                                                         dtr['test_recall'].mean(),
                                                         dtr['test_f1_score'].mean()],
                                        
                                        'Random Forest':[rfc['test_accuracy'].mean(),
                                                         rfc['test_precision'].mean(),
                                                         rfc['test_recall'].mean(),
                                                         rfc['test_f1_score'].mean()],
                                        
                                        'Gaussian Naive Bayes':[gnb['test_accuracy'].mean(),
                                                         gnb['test_precision'].mean(),
                                                         gnb['test_recall'].mean(),
                                                         gnb['test_f1_score'].mean()]},
                                        
                                        index=['Accuracy','Precision','Recall','F1 Score'])
    
    # Add 'Best Score column'
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    
    # Return models performance metrics scores data frame
    return models_scores_table


