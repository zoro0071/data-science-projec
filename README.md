AnomaData (Automated Anomaly Detection for Predictive Maintenance)

PROJECT OVERVIEW
AnomaData is a data science project aimed at predicting machine breakdown by identifying anomalies in the data. 
The project uses time series data collected from sensors installed in machines to detect patterns that indicate potential breakdowns. 
By predicting these anomalies early, maintenance can be scheduled proactively, reducing downtime and maintenance costs.

PROBLEM STATEMENT
The main objective of this project is to develop a machine learning model that can accurately predict anomalies in machine data. 
The model should be able to analyze the data collected from sensors and identify patterns that indicate potential breakdowns. 
The goal is to minimize false positives and false negatives to ensure timely and accurate predictions.

DATA SOURCES
The data used in this project is collected from sensors installed in machines. The dataset contains several features, including sensor readings, timestamps, and machine identifiers.
The data is collected at regular intervals and spans several months.

METHODOLOGY
The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, which consists of the following steps:

Business Understanding: Understanding the business objectives and requirements.
Data Understanding: Exploring and understanding the dataset.
Data Preparation: Preprocessing the data for model training.
Modeling: Building and training the machine learning model.
Evaluation: Evaluating the model's performance.

MODEL TRAINING AND EVALUATION
The project uses a variety of machine learning algorithms, including NAIVE,MOVINGAVERAGE,AUTOREGRESSION,SARIMA,SARIMAX,EXPONENTIAL SMOOOTHING to train the anomaly detection model.
The models are evaluated using MEAN ABSOLUTE ERROR AS IT METRIC FOR METRIC EVALUATION

DATA PREPROCESSING
Before training the model, the data undergoes several preprocessing steps, including handling missing values, scaling features.
Additionally, the data is split into training and test sets to evaluate the model's performance.

DEPENDENCIES
The project requires the following dependencies:

Python 3.x
NumPy
Pandas
Scikit-learn
Matplotlib
statsmodels


SETUP AND RUNNING THE PROJECT

STEP 1. To set up and run the project, follow these steps:

Clone the repository to your local machine:
git clone https://github.com/zoro0071/data-science-projec.git

STEP 2. Install the required dependencies:
pip install -r requirements.txt

STEP 3. Run the Jupyter Notebook or Python script to train the model and evaluate its performance:
jupyter notebook anomadata.ipynb

STEP 4. Follow the instructions in the notebook/script to preprocess the data, train the model, and evaluate its performance.
The code will run each time if run in the same steps as in cells of jupyter notebook.

CONCLUSION
AnomaData is a comprehensive data science project that aims to predict machine breakdowns by detecting anomalies in sensor data. 
By proactively identifying potential issues, the project helps reduce downtime and maintenance costs, ultimately improving machine reliability and performance.
