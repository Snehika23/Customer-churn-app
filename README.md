📡 Customer Churn Prediction

A data analytics project exploring customer churn behaviour in the telecom 
industry — combining exploratory data analysis, visualisation, and predictive 
modelling to uncover what drives customers to leave, and building an interactive 
app to make those insights accessible to anyone.

🌐 Live App

👉 Open the Streamlit App

No installation needed — open the link and explore directly in your browser.

💡 What This Project Does

Telecom companies lose revenue every time a customer cancels their subscription. This project builds a system that:

Analyses customer behaviour patterns from historical data
Identifies which customers are at high risk of churning
Gives an instant churn probability for any customer profile
Shows which factors are driving that prediction
🗂 Project Structure
Customer-churn-app/
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── Telco-Customer-Churn.csv   # Dataset (7,043 customer records)
└── README.md
📊 Dataset

The project uses the Telco Customer Churn dataset — 7,043 real customer records with 21 features covering:

Demographics — gender, age group, dependents, partner
Services — phone, internet, streaming, security, tech support
Billing — contract type, payment method, monthly and total charges
Target — whether the customer churned (Yes / No)
⚙️ How It Works
1. Exploratory Data Analysis

The app lets you explore the dataset visually — distribution plots for numerical features, countplots for categorical ones, and a correlation heatmap.

2. Data Preprocessing
Missing values in TotalCharges replaced with 0
Target column encoded (Yes → 1, No → 0)
All categorical columns label-encoded
3. Handling Class Imbalance

The dataset has significantly more non-churn records than churn records. SMOTE (Synthetic Minority Oversampling Technique) is applied to balance the training data before model fitting.

4. Model Training

A Random Forest Classifier is trained with 5-fold cross-validation. You can adjust:

Number of trees
Test split size
Random state
Cross-validation folds
5. Prediction

Enter any customer's details and get an instant prediction:

Churn or No Churn verdict
Churn probability percentage
Visual probability bar
6. Batch Prediction

Upload a CSV of multiple customers and download a results file with predictions for all of them at once.

🛠 Tech Stack
Tool	Purpose
Python	Core language
Pandas, NumPy	Data manipulation
Matplotlib, Seaborn	Data visualisation
scikit-learn	Model training and evaluation
imbalanced-learn	SMOTE oversampling
Streamlit	Web app deployment
📈 Model Performance

The Random Forest model achieves approximately 80% test accuracy with balanced precision and recall across both classes after SMOTE resampling.

Key findings from feature importance analysis:

Tenure — customers with shorter tenure are much more likely to churn
Contract type — month-to-month customers churn at a significantly higher rate
Monthly charges — higher charges correlate with increased churn risk
Internet service type — fiber optic customers show elevated churn

## 👩‍💻 About Me

**Snehika** | Aspiring Data Analyst

This project is part of my data analytics portfolio, where I apply end-to-end skills across the full data lifecycle — from raw data cleaning and exploratory analysis to machine learning and interactive dashboard deployment.

I built this to demonstrate:
- Working with real-world messy datasets
- Extracting insights through EDA and visualisation
- Applying ML to solve a business problem (customer retention)
- Communicating results through a live, shareable web app

📫 Open to Data Analyst opportunities — feel free to connect!
