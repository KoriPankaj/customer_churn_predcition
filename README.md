# Customer_churn_predcition
## Problem Statement
At Sunbase company, we prioritize understanding our customers and ensuring their satisfaction. To achieve this,
we need to develop a machine learning model that predicts customer churn.

# Solution Proposed
Develop a machine learning model to predict customer churn based on historical customer data. 

# Tech Stack Used
Python
Streamlit
Machine learning algorithms
Github
Streamlit community cloud

# Data Loading and Basic EDA (Exploratory Data Analysis):
The code loads data from an Excel file using Pandas.
Basic information about the dataset, such as its shape, columns, and missing values, is displayed.
The dataset is examined for duplicates.
# Data Preprocessing:
Unnecessary columns (CustomerID, Name) are dropped.
The distribution of the target variable Churn is checked.
# Outlier Detection:
Continuous variables (Age, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB) are examined for outliers using box plots and histograms.
# Feature Engineering:
Average monthly data usage is calculated as a new feature.
Age groups are created and added as a new feature.
The correlation between the categorical column Gender and the target variable Churn is checked using the chi-squared test.
Columns are rearranged for better organization.
# Input and Output Split:
The predictors (features) and the target variable (Churn) are separated.
The dataset is split into training and testing sets using the train_test_split function.
# Encoding Categorical Variables:
The categorical variables Location and Age_Group are one-hot encoded using the ColumnTransformer and OneHotEncoder.
The encoded data is transformed into a DataFrame.
# Feature Scaling:
The encoded DataFrame columns are scaled using the MinMaxScaler through another ColumnTransformer.
The scaled data is transformed into a DataFrame.
# Automated Model Selection with TPOT:
The TPOT Classifier is used to automatically search for the best machine learning model and its hyperparameters.
TPOT runs for a specified number of generations and population size, optimizing for accuracy.
The best model is exported to a Python file (best_model.py).
# Manual Model Building:
A KNeighborsClassifier with specified hyperparameters is manually chosen as the best model.
A pipeline is created with preprocessing transformers (trf2 and trf3) and the chosen KNeighborsClassifier (best_model).
The pipeline is trained on the training data.
# Model Evaluation and Reporting:
The trained pipeline is used to predict outcomes on the test set.
Accuracy scores for both test and training sets are calculated.
A classification report is generated, showing precision, recall, and F1-score for each class.

# Tried Other Models for comparison

**Random Forest Classifier:**

A RandomForestClassifier is initialized.
A pipeline (model_pipeline_1) is created with preprocessing transformers (trf2 and trf3) and the RandomForestClassifier.
Predictions are made on the test set using the trained model.
A classification report is generated and displayed.

**MLP Classifier (Neural Network):**

An MLPClassifier (neural network) with specified architecture is created.
A pipeline (model_pipeline_2) is created with preprocessing transformers and the MLPClassifier.
Predictions are made on the test set using the trained model.
A classification report is generated and displayed.

**Light Gradient Boosting (LGBM):**

LightGBM model is initialized with specific hyperparameters.
A pipeline (model_pipeline_3) is created with preprocessing transformers and the LGBMClassifier.
Predictions are made on the test set using the trained model.
A classification report is generated and displayed.

Fine-Tuning LightGBM using GridSearchCV:

A parameter grid is defined for hyperparameter tuning.
A LightGBM model with default hyperparameters is initialized.
GridSearchCV is used to search for the best hyperparameters using cross-validation.
A pipeline (model_pipeline_3H) is created with preprocessing transformers and the GridSearchCV instance.
The pipeline is fitted to the training data.
The best model and its parameters are displayed.

**Artificial Neural Network (ANN):**

A Sequential neural network model is defined using Keras.
The model architecture consists of input, hidden, and output layers.
The model is compiled with binary cross-entropy loss and Adam optimizer.
Data is preprocessed using the transformers (trf2 and trf3) and converted to the appropriate data type.
The model is trained on the training data.
Predictions are made on the test set using the trained model.
Predicted probabilities are thresholded to binary labels.
The accuracy of the model is calculated and printed.

**Model Dumping:**

The KNeighborsClassifier model, which showed the best performance among all ML algorithms, is serialized and saved using the pickle module.

# Create a GitHub Repository:

Create a new repository on GitHub where you'll store your code and other related files.
Clone the repository to your local machine using git clone.

# Organize Your Repository:

Create folders for different sections of project, such as code, data, and models.
Place your code files, data files, and serialized model files in the respective folders.

# Create a Streamlit App:

Create a Streamlit app file (e.g., app.py) that loads your serialized model and implements the user interface.
Place this file in the code folder of your repository.

# Push Your Code to GitHub:

Use git add to stage your changes.
Use git commit -m "Initial commit" to commit your changes.
Use git push to push your changes to your GitHub repository.

# Deploy to Streamlit CLoud:

Connect your GitHub repository to Streamlit Sharing.
Configure the deployment settings, such as specifying the app file (app.py), environment setup, etc.

Once the deployment is complete, Streamlit Sharing will provide  with a URL for our deployed app.

