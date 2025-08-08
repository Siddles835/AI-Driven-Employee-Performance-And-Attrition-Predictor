# Employee Attrition Prediction: See the Signs Before they Resign

**1. Executive Summary**
This project developed a machine learning model to predict which employees are most likely to leave the company. By analyzing historical HR data, the model identifies employees at high risk of attrition, allowing the business to implement targeted retention strategies. This proactive approach aims to reduce turnover costs and improve overall workforce stability.

**2. Project Goals**
The primary goals of this project were to:

Build a predictive tool to identify employees at risk of leaving.

Prioritize retention efforts by focusing on the most vulnerable individuals.

Identify key factors that drive attrition to inform long-term HR strategy.

**3. Data & Methodology**
The model was built using a dataset of 1,470 employee records with 35 features, including demographics, job details, compensation, and work-life factors.

Key Findings: Initial analysis revealed a significant link between attrition and overtime hours, lower job satisfaction, and lower monthly income. It also highlighted a major class imbalance, with only 16% of the dataset representing employees who had left.

Preprocessing: To prepare the data, we used one-hot encoding for categorical features, standardized numerical data, and applied the SMOTE technique to balance the imbalanced dataset.

Modeling: We trained and evaluated several models, ultimately selecting a high-performing XGBoost (Extreme Gradient Boosting) model.

**4. Results & Key Insights**
The XGBoost model was the top-performing model, demonstrating superior ability to accurately identify potential leavers.

Model

Precision

Recall

F1-Score

ROC-AUC

XGBoost

0.87

0.62

0.72

0.85

The model's analysis of feature importance revealed the most significant predictors of attrition:

OverTime: Employees working overtime are significantly more likely to leave.

JobSatisfaction: Low job satisfaction is a major indicator of attrition risk.

MonthlyIncome: Income level is a key factor, particularly in lower salary brackets.

Age: Younger employees show a higher rate of turnover.

YearsAtCompany: Attrition is more common in the first two years of employment.

**5. Getting Started**
To clone this repository and run the project locally, follow these steps:

Clone the repository:

git clone https://github.com/your-username/attrition-prediction.git

Navigate to the project directory:

cd attrition-prediction

Set up a virtual environment (recommended):

python -m venv venv

Activate the virtual environment:

On Windows:

venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

Install the required libraries:

pip install -r requirements.txt

Run the main script: This script will train the model, evaluate its performance, and generate a report.

python main.py

**6. Conclusion & Next Steps**
This project successfully created a high-performing model that provides valuable insights into employee attrition. The findings suggest that focusing on work-life balance (reducing overtime), improving job satisfaction, and addressing compensation are the most impactful retention strategies.

Recommended next steps include:

Integrating the model into a live dashboard for real-time monitoring.

Enriching the model with additional data, such as employee survey results.

Conducting A/B tests to measure the effectiveness of new retention programs.

**Created by: Sidhaanth Kapoor (Siddles835)**
