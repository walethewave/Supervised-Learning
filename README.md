﻿# Supervised-Learning- Customer Segmentation and Predictive Modeling


## Project Overview
This project focuses on customer segmentation using **K-Means clustering** and building **classification models** to predict customer segments. The analysis provides actionable insights for targeted marketing strategies, customer engagement, and resource optimization.

---

## Explanation and Detailed Analysis

### 1. What Was Done
The project involved the following steps:

#### Clustering Analysis:
- Performed **K-Means clustering** to group customers based on their financial behavior and spending patterns.
- Evaluated clusters using:
  - **Silhouette Score**: Measures cluster compactness and separation (higher is better).
  - **Calinski-Harabasz (CH) Score**: Measures between-cluster variance relative to within-cluster variance (higher is better).
- Identified the optimal number of clusters (\(k\)) using these metrics.

#### Feature Analysis for Clusters:
- Normalized features like `BALANCE`, `PURCHASES`, and `CREDIT_LIMIT` to remove scale effects.
- Analyzed cluster-specific behaviors in terms of spending, payment, and account usage.

#### Classification Models:
- Trained machine learning models to classify customers into clusters:
  - Algorithms: **Random Forest**, **Decision Tree**, **AdaBoost**, and **Gradient Boosting**.
- Split data into training and testing sets to evaluate model performance.
- Reported accuracy scores for each classifier.

---

### 2. Purpose of the Analysis
- **Identify Customer Segments**: Group customers based on financial behavior and spending patterns.
- **Understand Behavior Patterns**: Derive actionable insights for marketing and product strategies.
- **Build Predictive Models**: Use machine learning to classify new customers into existing clusters.

---

### 3. Results and Interpretation

#### Clustering Evaluation
- Optimal \(k = 3\) based on **Silhouette Score** (\(0.250\)), indicating well-defined clusters.
- CH Score favored \(k = 2\) (\(1705.79\)), but \(k = 3\) was chosen for better cluster compactness.

| \(k\) | Silhouette Score | CH Score    |
|------|------------------|------------|
| 2    | 0.209589         | 1705.788428 |
| 3    | 0.250554         | 1604.861125 |
| 4    | 0.197669         | 1597.750997 |

#### Cluster Characteristics
- **Cluster 0 (Low Spenders)**:
  - Low `BALANCE`, `PURCHASES`, and minimal `CASH_ADVANCE`.
  - Represents cost-sensitive or inactive users.
- **Cluster 1 (High Spenders)**:
  - High `PURCHASES` (one-off and installment), high `CREDIT_LIMIT`, and regular payments.
  - Profitable customers with frequent and significant transactions.
- **Cluster 2 (Cash Advance Users)**:
  - High `CASH_ADVANCE` and frequency.
  - Customers relying on cash loans.

#### Classification Models
The cluster labels (0, 1, 2) were used as the target variable. Model performance:

| Model              | Accuracy Score |
|--------------------|----------------|
| Random Forest      | 96.37%         |
| Decision Tree      | 93.68%         |
| AdaBoost           | 95.70%         |
| Gradient Boosting  | **96.98%**     |

- **Best Model**: Gradient Boosting achieved the highest accuracy, effectively predicting clusters.

---

### 4. Conclusion
The project revealed three distinct customer segments:
1. **Low Spenders**: Cost-sensitive or inactive users.
2. **High Spenders**: Profitable, frequent users.
3. **Cash Advance Users**: Customers reliant on cash loans.

#### Key Insights:
- **Marketing Strategies**:
  - Target **Cluster 0** with incentives to increase engagement.
  - Retain **Cluster 1** with rewards programs.
  - Offer financial education or alternative credit products for **Cluster 2**.
- **Model Performance**:
  - The models demonstrated high predictive accuracy, showing the effectiveness of clustering for segmentation.

#### Next Steps:
- Perform feature importance analysis on models.
- Explore reasons for high reliance on cash advances in **Cluster 2**.
- Design tailored marketing campaigns based on cluster behaviors.

---

## Dataset Description
The dataset contains attributes related to customer credit card usage and behavior:

| Feature Name                | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `CUST_ID`                   | Unique customer identifier.                                                |
| `BALANCE`                   | Monthly average balance over the past 12 months.                          |
| `PURCHASES`                 | Total purchase amount in the last 12 months.                              |
| `ONEOFF_PURCHASES`          | Total amount spent on one-off purchases.                                   |
| `INSTALLMENTS_PURCHASES`    | Total amount spent on installment purchases.                               |
| `CASH_ADVANCE`              | Total cash advance amount.                                                 |
| `CREDIT_LIMIT`              | Credit limit assigned to the customer.                                     |
| `PAYMENTS`                  | Total payments made to reduce statement balance.                          |
| `MINIMUM_PAYMENTS`          | Total minimum payments due during the period.                             |
| `TENURE`                    | Number of months the customer has held the credit card.                   |

---

## Technologies Used
- **Python Libraries**:
  - `scikit-learn` (clustering and classification)
  - `pandas`, `numpy` (data processing)
  - `matplotlib`, `seaborn` (visualization)
- **Machine Learning Models**:
  - K-Means Clustering
  - Random Forest, Decision Tree, AdaBoost, Gradient Boosting

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/customer-segmentation.git
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script to execute the analysis.

---

## Acknowledgements
This project demonstrates the application of unsupervised and supervised machine learning techniques to uncover customer behavior patterns and enhance business decision-making. Special thanks to the contributors and community for their support.

---


```
