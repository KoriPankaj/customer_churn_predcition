# pip install pandas
# pip install scikit-learn



import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import chi2_contingency


from tpot import TPOTClassifier 
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_excel(r"E:\sunbase\customer_churn_large_dataset.xlsx")

# Data Preprocessing

# Basic Eda
data.shape
data.columns
data.info()
data.isnull().sum()
data.describe()
data.duplicated().sum() 
           
df =  data.drop(columns=['CustomerID', 'Name'])
df['Churn'].value_counts(normalize = True) # Balanced Dataset

# Detect outliers
con_var = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
plt.figure(figsize=(2, 50))
for i, col in enumerate(con_var, 1):
    plt.subplot(len(con_var),1, i)
    plt.boxplot(df[col])
    plt.title(col)

for col in con_var:
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    plt.hist(df[col], bins=20)  # You can adjust the number of bins as needed
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
    
    

# Feature Engineering

# we can take avg monthly usage as a feature 
# add avg monthly data usage
df['Avg Monthly_Data_Usage'] = df['Total_Usage_GB'] / df['Subscription_Length_Months']

# we can create age groups
age_bins = [0, 18, 30, 50, float('inf')]  # Age ranges
age_labels = ['Teenager', 'Young Adult', 'Adult', 'Senior']

# Add a new column for age groups
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
df = df.drop(columns = ['Age'])
plt.hist(df['Age_Group'], bins =20)

# checking feature importance
# correlation of categorical column with target column
contingency_table = pd.crosstab(df['Gender'], df['Churn'])
contingency_table
chi2 = chi2_contingency(contingency_table)
chi2 # 'Gender' doesn't have significant impact in target variable
df = df.drop(columns = ['Gender'])
# arrange columns in order
df = df[['Age_Group', 'Location', 'Subscription_Length_Months','Avg Monthly_Data_Usage', 'Monthly_Bill', 'Total_Usage_GB', 'Churn' ]]

# Input and Output Split
predictors = df.drop(columns=['Churn'])
target = df['Churn']

# Splitting data into training and testing data set
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.3, random_state=0)

# Encoding categorical variables
#x_train['Gender'].value_counts(normalize = True)
x_train['Location'].value_counts(normalize = True)
x_train['Age_Group'].value_counts(normalize = True)

trf2 = ColumnTransformer([('ohe', OneHotEncoder(sparse= False, handle_unknown='ignore'),['Location', 'Age_Group'])]
                         ,remainder= 'passthrough')
encod = trf2.fit_transform(x_train)
encod_df = pd.DataFrame(encod)

# Apply feature scaling
trf3 = ColumnTransformer([('scaler', MinMaxScaler(), encod_df.columns[:15])])
scaled = trf3.fit_transform(encod_df)
scaled_df = pd.DataFrame(scaled)

# Using TPOT classifier to findout the best model
tpot_clf = TPOTClassifier(generations=2, population_size=50, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
tpot_clf.fit(x_train, y_train)
tpot_clf.export('best_model.py')

#best model from Auto ML
best_model = KNeighborsClassifier(n_neighbors=18, p=1, weights="uniform")
#best_model.fit(x_train,y_train)

model_pipeline = make_pipeline(trf2,trf3,best_model)
model_pipeline.fit(x_train,y_train)

y_pred = pd.Series(model_pipeline.predict(x_test))
y_train_pred = pd.Series(model_pipeline.predict(x_train))

from sklearn.metrics import accuracy_score

test_score = accuracy_score(y_test, y_pred)
print(test_score)


train_score = accuracy_score(y_train, y_train_pred)
print(train_score)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred)) 

import pickle
pickle.dump(model_pipeline,open('model_pipeline1.pkl','wb'))

# Other MOdel Results 
################ Train Random Forest classifier ############################
model = RandomForestClassifier(random_state=42)

model_pipeline_1 = make_pipeline(trf2,trf3, model)
model_pipeline_1.fit(x_train, y_train)

# Make predictions on the test set
y_pred_1 = model_pipeline_1.predict(x_test)

# Evaluate the best model's performance
accuracy = accuracy_score(y_test, y_pred_1)
print(accuracy)
classification_rep = classification_report(y_test, y_pred_1)

print("Best Model Parameters:", grid_search.best_params_)
print("Best Model Accuracy:", accuracy)
print("\nClassification Report for Best Model:\n", classification_rep)

################### Train MLP classifier #################################

from sklearn.neural_network import MLPClassifier

model_2 = MLPClassifier(hidden_layer_sizes=(500,), max_iter=1000)

model_pipeline_2 = make_pipeline(trf2,trf3, model_2)

model_pipeline_2.fit(x_train, y_train)

# Make predictions on the test set
y_pred_2 = model_pipeline_2.predict(x_test)

# Evaluate the best model's performance
accuracy = accuracy_score(y_test, y_pred_2)
print(accuracy)
classification_rep = classification_report(y_test, y_pred_2)
print("Best Model Accuracy:", accuracy)
print("\nClassification Report for Best Model:\n", classification_rep)

################## Train Light Gradient Boosting ###############
import lightgbm as lgb

# LightGBM Model
lgb_model = lgb.LGBMClassifier( boosting_type='gbdt',objective='binary',num_leaves=2,max_depth=5,learning_rate=0.1,
                               n_estimators=500, random_state=42)


model_pipeline_3 = make_pipeline(trf2,trf3, lgb_model)

# Train the LightGBM model
model_pipeline_3.fit(x_train, y_train)

# Predict using the LightGBM model
# Make predictions on the test set
y_pred_3 = model_pipeline_3.predict(x_test)

# Evaluate the best model's performance
accuracy = accuracy_score(y_test, y_pred_3)
print(accuracy)
classification_rep = classification_report(y_test, y_pred_3)
print("Best Model Accuracy:", accuracy)
print("\nClassification Report for Best Model:\n", classification_rep)

### Fine the Model using Hyper parameters
# Define the parameter grid to search
param_grid = {
    'num_leaves': [15, 20, 25, 30],  # Experiment with different values
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
}

# Initialize the LightGBM model
lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(lgb_model, param_grid, cv=5, scoring='accuracy')

model_pipeline_3H = make_pipeline(trf2,trf3, grid_search)

# Fit the grid search to your data
model_pipeline_3H.fit(x_train, y_train)

# Get the best model after hyperparameter tuning
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

model_pipeline_LGBh =  make_pipeline(trf2,trf3, best_model)

y_pred_4 = model_pipeline_LGBh.predict(x_test)
# Evaluate the best model's performance
accuracy = accuracy_score(y_test, y_pred_4)
print("Best Model Parameters:", best_params)
print("Best Model Accuracy:", accuracy)


# Tried multiple modelbut all are giving around 50 % accuracy which shows model is a under fit model.

################## Train Artificial  Neural Network  ##########################

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Define the neural network architecture
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=x_train.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


x_train_encod = trf2.fit_transform(x_train)
x_train_scales = trf3.fit_transform(x_train_encod)
x_test_encod = trf2.fit_transform(x_test)
x_test_scales = trf3.fit_transform(x_test_encod)

x_train = x_train_scales.astype(np.float32)
x_test = x_test_scales.astype(np.float32)


# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

y_pred = model.predict(x_test_scales)
y_pred_binary = [1 if val > 0.5 else 0 for val in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print("Test Set Accuracy:", accuracy)

# dumping KNN classifer as it was the best one amongst all ML algorithms
import pickle
pickle.dump(model_pipeline,open('model_pipeline1.pkl','wb'))
