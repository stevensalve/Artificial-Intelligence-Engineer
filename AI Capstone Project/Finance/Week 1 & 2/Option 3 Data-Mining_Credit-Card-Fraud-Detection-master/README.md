## Data-Mining---Credit-Card-Fraud-Detection
### Project Description
Goal: To solve two main issues in credit card fraud detection - skewness of the data and cost-sensitivity

•	Applied Logistic Regression, Decision Tree, Random Forest and **Ensemble Learning** Methods with hyper-parameter tuning on **under-sampled and oversampled dataset**, and improved performance by 25% (Contributor: JoyWang0320)

•	Built Bayesian Minimum Risk models with fixed cost and real financial cost separately (**Cost-Sensitive Learning**), and improved performance by 35% (Contributor: huxxx412)
### Data
The dataset contains transactions made by credit cards in September 2013 by European cardholders, where we have **492 frauds out of 284,807 transactions**. The dataset is **highly unbalanced**, in which **the proportion of the fraud transactions is only 0.172%.**
**It contains 30 independent variables including 2 numerical features, ‘Time’ and ‘Amount’, and 28 principal components obtained with PCA.** Feature 'Time' is a timestamp, which means the seconds elapsed between each transaction and the first transaction in the dataset, and it is not included in our analysis. The feature 'Amount' is the transaction Amount, and this feature can be used for example-dependent cost-sensitive learning. **Feature class is the response variable and it takes value 1 for fraud transactions and 0 for normal ones. Finally, 29 independent variables and one dependent variable are included in our research.**

Link: https://www.kaggle.com/mlg-ulb/creditcardfraud
### Implementation Choice
Python Jupyter Notebook
