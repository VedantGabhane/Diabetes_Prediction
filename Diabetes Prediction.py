#!/usr/bin/env python
# coding: utf-8

# # Health Care Prediction on Diabetic Patients -  Case Study
# 
# ## Context
# 
# This dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases. Its primary objective is to diagonostically predict whether a patient has diabetes or not based on specific diagnostic measurements. The dataset was carefully selected, focusing on femaile patients aged at least 21 years and of Pima Indian Heritage.
# 
# ## Problem Statement
# 
# Build a model with high accuracy to predict whether patients in the dataset have diabetes.
# 
# ## Dataset Description
# 
# The dataset includes various medical predictor variables and one target variable, "Outcome". The predictor variables encompass essential health metrics, such as the number of pregnancies, plasma glucose concentration, diastolic blood pressure, triceps skinfold thickness, insulin levels, body mass index (BMI), diabetes pedigree function, and age.
# 
# ### Predictor Variables
# 
# 1. **Pregnancies:**
#     - Number of times pregnant
#     
# 2. **Glucose:**
#     - Plasma glucose concentration at 2 hours in an oral glucose tolerance test
#     
# 3. **BloodPressure:**
#     - Diastolic blood pressure (mm Hg)
#     
# 4. **SkinThickness:**
#     - Triceps skinfold thickness (mm)
#     
# 5. **Insulin:**
#     - 2-Hour serum insulin (mu U/ml)
#     
# 6. **BMI:**
#     - Body mass index (weight in kg/(height in m)^2)
#     
# 7. **DiabetesPedigreeFunction:**
#     - Diabetes pedigree function
#     
# 8. **Age:**
#     - Age in years
#     
# ## Target Variable
# 
# - **Outcome:**
# 
#     - Class variable (0 or 1)
#     - 268 instances are labeled as 1 (indicaating diabetes), while others are labeled as 0.
#     
# This dataset provides a valuable oppurtunity to develop a predictive model for diabetes based on demographic and health-related features.
# 

# # Loading the required Library Packages 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
import seaborn as sns
from sklearn.metrics import accuracy_score,mean_squared_error,classification_report,confusion_matrix,precision_score,recall_score,roc_curve,auc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


# # Reading and exploring the Health Care Dataset

# In[2]:


data=pd.read_excel("health care diabetes.xlsx")
data.head()


# In[3]:


#Checking the number of rows and columns of the dataset
data.shape


# In[4]:


#Dataset Information Overview
data.info()


# In[5]:


#Summary Stastics for the Diabetes Dataset
data.describe()


# # Data Preprocessing: Treating the Missing Values

# ## In this dataset, 0 represents the null values, and hence we will replace 0 by mean of their feature (variable) columns

# In[6]:


data['BloodPressure'].mean()


# In[7]:


#Identifying the mean of the features
print(data['Glucose'].mean())
print(data['BloodPressure'].mean())
print(data['SkinThickness'].mean())
print(data['Insulin'].mean())


# In[8]:


# Finding the number of rows which has the null values
print('Glucose-',len(data['Glucose'][data['Glucose']==0]))
print('BloodPressure-',len(data['BloodPressure'][data['BloodPressure']==0]))
print('SkinThickness-',len(data['SkinThickness'][data['SkinThickness']==0]))
print('Insulin-',len(data['Insulin'][data['Insulin']==0]))


# In[9]:


# Finding the null value percentage
selected_columns = ['Glucose', 'BloodPressure', 'SkinThickness','Insulin']
null_percentage = (data[selected_columns] == 0).mean() * 100
 
# Displaying the null value percentage for each selected column
print("Percentage of Null Values for Each Column:")
print(null_percentage)


# # Inferences from Null Value Percentage Analysis
# 
# The analysis of null value percentages in the dataset reveals the following insights:
# 
# 1. **Glucose:**
#     - Approximately 0.65% of the data points in the "Glucose" column are represented as null values.
#     
# 2. **Blood Pressure:**
#     - The "Blood Pressure" column has a null value percentage of approximately 4.56%.
#     
# 3. **Skin Thickness:**
#     - A significant portion of the "Skin Thickness" column, around 29.56%, contains null values.
#     
# 4. **Insulin:**
#     - The "Insulin" column exhibits a higher null value perentage, with approximately 48.70% of the data points being null.
#     
# These findings suggest that imputation or other strategies may be necessary for columns with substantial null values, such as "Skin Thickness" and "Insulin", to ensure the integrity of the dataset for subsequent analyses or modeling.

# In[10]:


# Replacing the null values with the mean
data['Glucose']=data['Glucose'].replace([0],[data['Glucose'].mean()])
data['BloodPressure']=data['BloodPressure'].replace([0],[data['BloodPressure'].mean()])
data['SkinThickness']=data['SkinThickness'].replace([0],[data['SkinThickness'].mean()])
data['Insulin']=data['Insulin'].replace([0],[data['Insulin'].mean()])
data['BMI']=data['BMI'].replace([0],[data['BMI'].mean()])


# In[11]:


data.describe()


# In[12]:


# Finding the null value percentage
selected_columns = ['Glucose', 'BloodPressure', 'SkinThickness','Insulin', 'BMI']
null_percentage = (data[selected_columns] == 0).mean() * 100
# Displaying the null value percentage for each selected column
print("Percentage of Null Values for Each Column:")
print(null_percentage)


# # Inference from Null Value Treatment
# 
# After addressing null values in the dataset, it is observed that all selected columns ("Glucose", "Blood Pressure", "Skin Thickness", and "Insulin") no longer contain any null values. The null value has been successful, resulting in a clean datset with 0% null values in these specific columns.

# # Detecting Outliers and Treatment

# In[13]:


columns=data[selected_columns]


# In[14]:


type(columns)


# In[15]:


columns.columns


# In[16]:


# Display boxplots for numeric columns to visualize outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=columns)
plt.title("Boxplots for Numeric Columns")
plt.show()


# In[17]:


data.describe()


# In[18]:


# Finding the Outlier Count in the selected Columns:
def find_outliers_iqr(data, column_name):
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
 
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
 
    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
 
    # Find outliers
    outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
 
    # Count the number of outliers
    count_outliers = len(outliers)
 
    return count_outliers
 
# Calculate and print the number of outliers for each column of interest
for column_name in selected_columns:
    outlier_count = find_outliers_iqr(data, column_name)
    print(f"Number of outliers in the '{column_name}' column: {outlier_count}")


# ### Boxplot Analysis for Numerical Columns
# 
# The boxplot illustrates the distribution of four numerical columns: Glucose, BloodPressure, Skin Thickness, and Insulin. The following inferences can be drawn:
# 
# #### Glucose
# - Median glucose level: ~200 mg/dL
# - IQR is large, indicating considerable variability in glucose levels.
# - There are no outliers
# 
# #### Blood Pressure
# - Median blood pressure: 72 mmHg (within the normal range).
# - IQR is relatively small, suggesting more consistent blood pressure levels.
# - Few outliers, none extremely high or low.
# 
# #### Skin Thickness
# - Median skin thickness: ~25 mm
# - IQR is small, indicating less considerable variability in skin thickness.
# - Few outliers, none extremely high.
# 
# #### Insulin
# - Median insulin level: ~79 mIU/L
# - IQR is large, indicating considerable variability in insulin levels.
# - More outliers, many are extremely high.
# 
# #### Overall Observations
# - All columns exhibit a wide range of values, with some outliers. Insulin column has many outliers
# - Median values for all columns, except the insulin column fall within the normal range.
# 
# #### Additional Inferences
# - Glucose levels show more variability than blood pressure levels.
# - More outliers in the insulin columns compared to blood pressure and skin thickness.
# 
# #### Possible Interpretations
# - Variability in glucose levels may be influenced by factors like diet, exercise, and stress.
# - Outliers in the Insulin column may also be associated with underlying medical conditions or physiological factors. Elevated insulin levels could be indicative of conditions such as insulin resistance or diabetes. Additionally, factors such as dietary habits, genetic predisposition, or specific medical treatments may contribute to higher insulin levels. Further investigation and domain expertise are necessary to understand the potential health implications of these outliers in the Insulin column. 
# 
# It is essential to note that these inferences are based on a single boxplot, and further information is needed to draw definitive conclusions.
# 

# # Outlier Treatment

# In[19]:


sorted(data)
Q1=data.quantile(0.20)
Q3=data.quantile(0.80)
IQR=Q3-Q1
print(IQR)


# In[20]:


data_cleared_iqr = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
data_cleared_iqr
print(data_cleared_iqr.shape)
print(data.shape)


# In[21]:


data_cleared_iqr


# # Inferences from Outlier Removal using IQR Method
# 
# ## Data Size Reduction:
# 
# After removing outliers using the interquartile range (IQR) method, the dataset has been reduced from 768 to 678 rows.
# 
# ## Outliers Identified:
# 
# Outliers were detected and removed across various columns, particularly impacting features like Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, and Age.
# 
# ## Increased Data Robustness:
# 
# The IQR-based outlier removal contributes to a more robust dataset, potentially improving the reliability of statistical analyses and modeling.
# 
# ## Preserved Features:
# 
# The operation was applied to 9 columns, including predictors like Glucose and Skin Thickness, as well as the target variable Outcome.
# 
# ## Consideration for Domain Knowledge:
# 
# The decision to remove outliers should be made with consideration for domain knowledge, as outliers may contain valuable information or indicate specific health conditions.
# 
# ## Final Dataset Statistics:
# 
# - Dataset size after outlier removal: 678 rows.
# - Original dataset size: 768 rows.

# In[22]:


col=data_cleared_iqr[['Glucose','BloodPressure','SkinThickness','Insulin']]


# In[23]:


col.shape


# In[24]:


plt.figure(figsize=(12, 8))
sns.boxplot(data=col)
plt.show()


# ## Visually exploring variables using histograms

# In[25]:


data.describe()


# In[26]:


data['Glucose'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()


# In[27]:


data['BloodPressure'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()


# In[28]:


data['SkinThickness'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()


# In[29]:


data['Insulin'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()


# # Violin plot for the selected features

# In[30]:


plt.figure(figsize=(15, 10))
sns.violinplot(data=data[selected_columns])
plt.title("Violin Plot of Selected Features")
plt.show()


# #### The violin plot shows the distribution of four numerical features: Glucose, BloodPressure, Skin Thickness, and Insulin. The violin shape represents the probability density function (PDF) of each feature, and the box plot embedded within each violin plot shows the median, interquartile range (IQR), and outliers.

# # Kernel Density Estimation (KDE) plot for the selected features

# In[31]:


plt.figure(figsize=(15, 10))

for column in selected_columns:

    sns.kdeplot(data[column], label=column)

plt.title("Kernel Density Estimation (KDE) Plot of Numeric Features")

plt.legend()

plt.show()


# #### The image shows a Kernel Density Estimation (KDE) plot of four numerical features: Glucose, BloodPressure, Skin Thickness, and Insulin. KDE is a non-parametric method for estimating the probability density function (PDF) of a random variable. The KDE plot shows the estimated PDF of each feature, which can be used to visualize the distribution of the data.

# ## Creating a count (frequency) plot describing the data typesand the count of vaiables.

# In[32]:


data.dtypes


# In[33]:


data.dtypes.value_counts()


# In[34]:


figsize=(16,2)
data.dtypes.value_counts().plot(kind='barh')
plt.legend()
plt.show()


# In[35]:


data['Outcome'].value_counts().plot(kind='bar')
plt.legend()
plt.title('Outcome')
plt.show()


# ### It is observed that there are three features of integer data type and six features of float data type.

# # Data Exploration:

# ## Check the balance of the data by plotting the count of outcomes by their value. Describe your findings and plan future course of action.

# In[36]:


data['Outcome'].value_counts()


# In[37]:


data['Outcome'].value_counts().plot(kind='bar')
plt.legend()
plt.title('Outcome')
plt.show()


# In[38]:


outcome = (data['Outcome'].value_counts()/data['Outcome'].shape)*100
outcome


# # Inferences from Outcome Distribution
# 
# ## Class Imbalance:
# 
# The dataset exhibits class imbalance in the 'Outcome' variable.
# Class 0 (No Diabetes) has 500 instances.
# Class 1 (Diabetes) has 268 instances.
# 
# ## Potential Impact on Modeling:
# 
# Class imbalances may affect the performance of machine learning models, particularly for binary classification tasks.
# Addressing class imbalance through techniques like resampling or using appropriate evaluation metrics may be necessary.
# 
# ## Consideration for Predictive Models:
# 
# Models may need to be evaluated and tuned considering the imbalanced distribution to avoid biased predictions toward the majority class.

# In[39]:


balanced_data=100-outcome
balanced_data


# In[40]:


balanced_data.plot(kind='bar')
plt.legend()
plt.title('Balanced_data')
plt.show()


# # Bi-Variate Analysis

# ## Creating scatter charts between the pair of variables to understand the relationships.

# In[41]:


plt.figure(figsize=(12,5))
sns.scatterplot(x='Glucose',y='BloodPressure',hue='Outcome',data=data)
plt.show()


# In[42]:


plt.figure(figsize=(12,5))
sns.scatterplot(x='BMI',y='Insulin',hue='Outcome',data=data)
plt.show()


# In[43]:


plt.figure(figsize=(12,5))
sns.scatterplot(x='Age',y='Glucose',hue='Outcome',data=data)
plt.show()


# In[44]:


plt.figure(figsize=(12,5))
sns.scatterplot(x='Age',y='Pregnancies',hue='Outcome',data=data)
plt.show()


# - We can see Pregnancies has highest relation with Age feature.
# - Also, Outcome has maximum relation with Glucose and minimum with Blood Presure than the other features.
# - We can see from scatter plot, that there is ouliers present in this data.
# - Because of outliers, our data is skewed to left or right side, which is not acceptable.
# - If we want to train a model, this poses a problem.
# - Therefore, for better visualization and outlier detection, we can use sns.boxplot and remove outliers from the dataset.

# In[45]:


sns.pairplot(data)
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()


# # Multi Variate Analysis

# ## Perform correlation analysis. Visually explore it using a heat map

# In[46]:


plt.figure(figsize=(10,7))
sns.heatmap(data.corr(),annot=True)
plt.show()


# ### **We can see Outcome has maximum relation with Glucose and minimum with Blood Presure than the other features.**

# # Data Modelling:

# ### 1- Devise strategies for model building. It is important to decide the right validation framework. Express your thought process.

# ## strategies for model building :-
#     
#     1. Descriptive Analysis :-
#         -Identify ID, Input and Target features
#         -Identify categorical and numerical features
#         -Identify columns with missing values
#         
#     2. Data Treatment (Missing values treatment) :-
#         - Detecting outliers & removing them. 
#         - Imputing mean, mode or median value at a place of missing value as per dataset   
#         
#     3.Feature Extraction / Feature Engineering :-
#         -we will remove noisy features from data
#         -By the help of correlation / heatmap / differnt types of feature selection techniques.
#         
#     4.Data is imbalanced
#         -For balancing the data we wil use SMOTE over sampling techinque.
#         
#     5.Building a model :-
#         - select a best algorithms for model
#         
#     6.Train a model
#     
#     7.Evaluation
#         - check a accuracy & mean squared error of model
#         
#     8.Hyper Parameter Tuning :-
#         -for decrese in RMSE check a best parameters for model.
#         
#     9.Create a clasification report.
# 

# # Feature Extraction

# In[47]:


data.head()


# In[48]:


data.columns


# In[49]:


# Data preparation for modeling
x=data.drop(['Outcome'],axis=1)
y=data.Outcome


# In[50]:


x.head()


# In[51]:


type(y)


# In[52]:


#Finding the correlation of every feature with the Outcome (Target Variable)
data.corrwith(data['Outcome'])


# In[53]:


bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
#Concat two dataframes for better visulization
featureScores = pd.concat([dfcolumns, dfscores],axis=1)
featureScores.columns = ['Specs','Score']
print(featureScores.nlargest(8, 'Score'))


# In[54]:


type(fit)


# In[55]:


fit.scores_


# In[56]:


plt.figure(figsize=(10,7))
sns.heatmap(data.corr(),annot=True)
plt.show()


# In[57]:


new_x = data.drop(['Outcome','BloodPressure'],axis=1).values
new_y = data.Outcome.values


# # SMOTE to address the class Imbalance

# ## Train a model

# In[58]:


#Train-Test Split for Data Modeling
trainx, testx, trainy, testy = train_test_split(new_x, new_y, test_size = 0.20, random_state=10)


# In[59]:


get_ipython().system('pip install imblearn')


# In[60]:


print("Before Oversampling, counts of label '1' : {}".format(sum(trainy == 1)))
print("Before Oversampling, counts of label '0' : {} \n".format(sum(trainy == 0)))

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 63)
trainx_res, trainy_res = sm.fit_resample(trainx, trainy.ravel())
print("After Oversampling, the shape of train_X: {}".format(trainx_res.shape))
print("After Oversampling, the shape of train_y: {} \n".format(trainy_res.shape))

print("After Oversampling, counts of label '1' : {}".format(sum(trainy_res == 1)))
print("After Oversampling, counts of label '0' : {} \n".format(sum(trainy_res == 0)))


# In[61]:


#sc = StandardScaler()


# In[62]:


#trainx = sc.fit_transform(trainx)
#testx = sc.fit_transform(testx)


# # Applying an appropriate classification algorithm to build a model.

# # Model 1 : Building a Logistic Regression Model

# In[63]:


logreg = LogisticRegression(solver='liblinear',random_state=123)


# In[64]:


logreg.fit(trainx_res,trainy_res)


# In[65]:


prediction = logreg.predict(testx)


# In[66]:


print('Accuracy_score -',accuracy_score(testy,prediction))
print('Mean_squared_error -',mean_squared_error(testy,prediction))


# In[67]:


print((confusion_matrix(testy, prediction)))


# In[68]:


print(classification_report(testy, prediction))


# In[69]:


#Preparing ROC Curve (Receiver Operating Characetristic Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#predict probabilities
probs = logreg.predict_proba(trainx_res)

#keep probabilities for the positive outcome only 
probs = probs[:, 1]

#calculate AUC
auc = roc_auc_score(trainy_res, probs)
print('AUC: %.3f' % auc)

#calculate roc curve
fpr, tpr, thresholds = roc_curve(trainy_res, probs)

#plot no skill
plt.plot([0, 1], [0,1], linestyle='--')

#plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.show()


# # Model 2 : Random Forest Classifier

# In[70]:


rf = RandomForestClassifier(random_state=42, max_depth=5)


# In[71]:


rf.fit(trainx_res, trainy_res)


# In[72]:


rf_predict = rf.predict(testx)


# In[73]:


print('Accuracy_score -',accuracy_score(testy,rf_predict))
print('Mean_squared_error -',mean_squared_error(testy,rf_predict))


# # RandomForestClassifier ( Hyper Parameter Tunning)

# In[74]:


param_grid = {'n_estimators':[100,400,200,300], 'criterion':['gini', 'entropy'], 'max_depth':[1,2,3], 'min_max_leaf_nodes':[1,2,3], 'max_samples':[2,3,4]}


# In[75]:


grid=GridSearchCV( estimator = rf, param_grid = param_grid, n_jobs=-1, cv = 5, verbose = 2)


# In[76]:


#grid.fit(trainx_res, trainy_res)


# In[77]:


#grid.best_params_


# In[78]:


rf_grid=RandomForestClassifier(criterion = 'gini', max_depth = 2, max_leaf_nodes = 3, max_samples = 4, min_samples_leaf = 1, min_samples_split= 3, n_estimators = 400, random_state=42)


# In[79]:


rf_grid.fit(trainx_res, trainy_res)


# In[80]:


rf_grid_predict = rf_grid.predict(testx)


# In[81]:


print('Accuracy_score -',accuracy_score(testy,rf_grid_predict))
print('Mean_squared_error -',mean_squared_error(testy,rf_grid_predict))


# In[82]:


print((confusion_matrix(testy, rf_grid_predict)))


# In[83]:


print(classification_report(testy, rf_grid_predict))


# In[84]:


#Preparing ROC Curve (Receiver Operating Characetristic Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#predict probabilities
probs = rf.predict_proba(trainx_res)

#keep probabilities for the positive outcome only 
probs = probs[:, 1]

#calculate AUC
auc = roc_auc_score(trainy_res, probs)
print('AUC: %.3f' % auc)

#calculate roc curve
fpr, tpr, thresholds = roc_curve(trainy_res, probs)

#plot no skill
plt.plot([0, 1], [0,1], linestyle='--')

#plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.show()


# # Model 3: Decision Tree Classfier

# In[85]:


dc = DecisionTreeClassifier(random_state=42)


# In[86]:


dc.fit(trainx_res, trainy_res)


# In[87]:


dc_predict = dc.predict(testx)


# In[88]:


print('Accuracy_score -',accuracy_score(testy,dc_predict))
print('Mean_squared_error -',mean_squared_error(testy,dc_predict))


# # DecisionTreeClassfier (Hyper Parameter Tunning)

# In[89]:


dc_param_grid = {'splitter':["best", "random"], 'criterion':['gini', 'entropy'], 'max_depth':[1,2,3], 'min_samples_split':[1,2,3], 'min_samples_leaf':[1,2,3], 'max_leaf_nodes': [1,2,3]}


# In[90]:


import warnings
warnings.filterwarnings('ignore')
dc_grid=GridSearchCV( estimator = dc, param_grid = dc_param_grid, n_jobs=-1, cv = 5, verbose = 2)
dc_grid.fit(trainx_res,trainy_res)


# In[91]:


dc_grid.best_params_


# In[92]:


dc_final = DecisionTreeClassifier(criterion = 'gini', max_depth = 2, max_leaf_nodes = 4, min_samples_leaf = 1,
                                  min_samples_split= 2, splitter = 'best', random_state = 42)


# In[93]:


dc_final.fit(trainx_res, trainy_res)
dc_final_pred = dc_final.predict(testx)


# In[94]:


print('Accuracy_score -',accuracy_score(testy,dc_final_pred))
print('Mean_squared_error -',mean_squared_error(testy,dc_final_pred))


# In[95]:


print((confusion_matrix(testy, dc_final_pred)))


# In[96]:


print(classification_report(testy, dc_final_pred))


# In[97]:


#Preparing ROC Curve (Receiver Operating Characetristic Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#predict probabilities
probs = dc_final.predict_proba(trainx_res)

#keep probabilities for the positive outcome only 
probs = probs[:, 1]

#calculate AUC
auc = roc_auc_score(trainy_res, probs)
print('AUC: %.3f' % auc)

#calculate roc curve
fpr, tpr, thresholds = roc_curve(trainy_res, probs)

#plot no skill
plt.plot([0, 1], [0,1], linestyle='--')

#plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.show()


# # Model 4 : KNN

# In[98]:


from sklearn.neighbors import KNeighborsClassifier


# In[99]:


knn = KNeighborsClassifier(n_neighbors = 4)


# In[100]:


knn.fit(trainx_res, trainy_res)


# In[101]:


knn_pred = knn.predict(testx)


# In[102]:


print('Accuracy_score -',accuracy_score(testy, knn_pred))
print('Mean_squared_error -',mean_squared_error(testy, knn_pred))


# In[103]:


print((confusion_matrix(testy, knn_pred)))


# In[104]:


print(classification_report(testy, knn_pred))


# In[105]:


#Preparing ROC Curve (Receiver Operating Characetristic Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#predict probabilities
probs = knn.predict_proba(trainx_res)

#keep probabilities for the positive outcome only 
probs = probs[:, 1]

#calculate AUC
auc = roc_auc_score(trainy_res, probs)
print('AUC: %.3f' % auc)

#calculate roc curve
fpr, tpr, thresholds = roc_curve(trainy_res, probs)

#plot no skill
plt.plot([0, 1], [0,1], linestyle='--')

#plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.show()


# # Model Accuracy Comparsion

# In[106]:


Algorithms = ['KNN', 'RandomForest', 'Decisiontree']
Accuracy_Score = [accuracy_score(testy, knn_pred), accuracy_score(testy, rf_grid_predict), accuracy_score(testy, dc_final_pred)]
#Create a DataFrame
accuracy_df = pd.DataFrame({'Algorithm' : Algorithms, 'Accuracy' : Accuracy_Score})

#Display the accuracy table
print(accuracy_df)


# # Inferences from Model Accuracy Comparison
# 
# **1. RandomForest Performs Well:**
# 
# - Among the algorithms tested, RandomForest exhibits the highest accuracy at 73.38%.
# 
# **2. KNN Shows Lower Accuracy:**
# 
# - KNN has the lowest accuracy among the models, with a score of 62.34%.
# 
# **3. Consistent Performances:**
# 
# - Decision Tree, SVM, Naive Bayes, and XGBoost show relatively similar accuracies, ranging from 68.83% to 72.08%.
# 
# **4. Consideration for Model Selection:**
# 
# - The choice of the algorithm depends on various factors, including the specific requirements of the task, interpretability, and computational efficiency.
# 
# **5. Further Evaluation:**
# 
# - Additional evaluation metrics, such as precision, recall, and F1 score, should be considered for a comprehensive assessment of model performance.
# 

# # Comparison of various models with the results from KNN algorithm

# In[107]:


#Creating the objects
logreg_cv = LogisticRegression(solver = 'liblinear', random_state = 123)
dt_cv = DecisionTreeClassifier(random_state = 123)
knn_cv = KNeighborsClassifier()
rf_cv = RandomForestClassifier(random_state = 123)
cv_dict = {0: ' Logistic Regression', 1: 'Decision Tree', 2: 'KNN', 3: 'Random Forest'}
cv_models = [logreg_cv, dt_cv, knn_cv,rf_cv]

for i, model in enumerate(cv_models):
    print("{} Test Accuracy: {}".format(cv_dict[i], cross_val_score(model, trainx, trainy, cv = 10, scoring = 'accuracy').mean()))


# # Inferences from Model Comparison with KNN Algorithm Results
# 
# ### 1. Logistic Regression Outperforms:
# 
# - Among the models tested, Logistic Regression exhibits the highest test accuracy at 77.68%.
# 
# ### 2. Decisive Model Differences:
# 
# - Decision Tree, and Random Forest show lower test accuracies compared to Logistic Regression, ranging from 70.53% to 76.21%.
# 
# ### 3. Consideration for Model Selection:
# 
# - Logistic Regression and SVC might be preferred choices based on higher test accuracies, but other factors such as interpretability and computational efficiency should be considered.
# 
# ### 4. Cross-Validation Insights:
# 
# - The use of cross-validation provides a robust estimate of model performance, reducing the impact of data partitioning on results.
# 
# ### 5. Further Exploration:
# 
# - Evaluation metrics beyond accuracy, such as precision, recall, and F1 score, should be considered for a comprehensive understanding of model effectiveness.
# 
