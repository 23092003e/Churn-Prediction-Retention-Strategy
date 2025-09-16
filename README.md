# 📊 Customer Churn Prediction & Retention Strategy

A comprehensive data science project that analyzes telecom customer churn patterns, builds predictive models, and provides actionable retention insights through advanced analytics and Power BI visualization.

## 🎯 Project Overview

This project delivers end-to-end customer churn analysis for a telecommunications company, combining exploratory data analysis, machine learning modeling, and business intelligence dashboards to drive customer retention strategies.

**Key Deliverables:**
- Data cleaning and preprocessing pipeline
- Comprehensive exploratory data analysis (EDA)
- Multiple machine learning models for churn prediction
- Power BI dashboard with interactive visualizations
- Business insights and retention recommendations

## 📁 Project Structure

```
Churn-Prediction-Retention-Strategy/
├── Customer_Churn_Analysis.ipynb    # Main analysis notebook
├── Dashboard.pbix                   # Power BI dashboard
├── churn_prediction_model.pkl       # Trained ML model (21MB)
├── customer_data_cleaned.csv        # Processed dataset
├── data/
│   └── customer_data.csv           # Original raw dataset
└── README.md                       # Project documentation
```

## 🔍 Analysis Components

### 1. Data Understanding & Cleaning
- **Dataset**: 7,032 telecom customers with 33 features
- **Target Variable**: Customer churn status (Churned/Stayed)
- **Data Quality**: Missing value handling, text normalization, outlier detection
- **Feature Engineering**: Revenue calculations, tenure grouping, service bundling

### 2. Exploratory Data Analysis (EDA)
- **Churn Rate Analysis**: Overall churn rate and distribution patterns
- **Demographic Insights**: Age, gender, and location impact on churn
- **Service Analysis**: Contract types, internet services, and add-on features
- **Financial Analysis**: Revenue patterns, monthly charges, and customer lifetime value
- **Correlation Analysis**: Feature relationships and multicollinearity detection

### 3. Machine Learning Models
- **Baseline Models**: Logistic Regression, Random Forest
- **Advanced Models**: Gradient Boosting, SVM
- **Ensemble Methods**: Voting Classifier, Stacking Classifier
- **Model Evaluation**: Accuracy, ROC-AUC, Precision, Recall, F1-Score
- **Feature Importance**: Identification of key churn predictors

### 4. Business Intelligence Dashboard
- **Power BI Visualization**: Interactive dashboard with KPIs
- **Customer Segmentation**: Risk-based customer categorization
- **Revenue Impact**: Financial loss analysis from churn
- **Retention Insights**: Actionable recommendations for customer retention

## 📊 Key Findings

### Churn Patterns
- **Month-to-Month Contracts**: Highest churn risk segment
- **Fiber Optic Internet**: Higher churn rates compared to cable/DSL
- **New Customers**: Higher churn in first 12 months
- **Service Bundles**: Customers with fewer services show higher churn

### Financial Impact
- **Revenue at Risk**: Quantified potential revenue loss from churn
- **Customer Lifetime Value**: Analysis of high-value customer segments
- **Retention ROI**: Cost-benefit analysis of retention strategies

### Predictive Performance
- **Model Accuracy**: 85%+ accuracy on test data
- **ROC-AUC Score**: 0.90+ for ensemble models
- **Feature Importance**: Contract type, tenure, and monthly charges as top predictors

## 🛠️ Technical Stack

- **Programming**: Python 3.x
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Model Persistence**: Joblib
- **Business Intelligence**: Microsoft Power BI
- **Environment**: Jupyter Notebook

## 📋 Requirements

```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Churn-Prediction-Retention-Strategy
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

3. **Run the analysis**
   ```bash
   jupyter notebook Customer_Churn_Analysis.ipynb
   ```

4. **View Power BI Dashboard**
   - Open `Dashboard.pbix` in Microsoft Power BI Desktop
   - Explore interactive visualizations and KPIs

## 📈 Model Performance

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|-------|----------|---------|-----------|--------|----------|
| Logistic Regression | 82.1% | 0.876 | 0.84 | 0.78 | 0.81 |
| Random Forest | 85.3% | 0.901 | 0.87 | 0.82 | 0.84 |
| Gradient Boosting | 86.1% | 0.912 | 0.88 | 0.83 | 0.85 |
| Ensemble (Stacking) | 87.2% | 0.924 | 0.89 | 0.85 | 0.87 |

## 💡 Business Recommendations

### High-Priority Actions
1. **Contract Optimization**: Incentivize long-term contracts for month-to-month customers
2. **Early Intervention**: Implement retention programs for customers in first 12 months
3. **Service Bundling**: Promote additional services to increase customer stickiness
4. **Fiber Optic Support**: Improve customer experience for fiber optic internet users

### Retention Strategies
- **Personalized Offers**: Target high-risk customers with customized retention offers
- **Customer Success Programs**: Proactive outreach for at-risk segments
- **Product Improvements**: Address pain points identified in churn reasons
- **Loyalty Programs**: Reward long-term customers to reduce churn propensity

## 📊 Dashboard Features

The Power BI dashboard includes:
- **Executive Summary**: High-level KPIs and metrics
- **Churn Analysis**: Detailed churn patterns and trends
- **Customer Segmentation**: Risk-based customer categorization
- **Revenue Impact**: Financial analysis of churn impact
- **Predictive Insights**: Model predictions and feature importance

## 🔄 Model Deployment

The trained model (`churn_prediction_model.pkl`) can be deployed for:
- **Batch Scoring**: Monthly churn risk assessment
- **Real-time Prediction**: API-based churn prediction service
- **Automated Alerts**: Proactive identification of at-risk customers

## 📞 Contact & Support

For questions about this analysis or collaboration opportunities:
- Review the Jupyter notebook for detailed methodology
- Examine the Power BI dashboard for business insights
- Check model performance metrics in the analysis results

## 📄 License

This project is available for educational and research purposes. Please ensure appropriate data privacy and ethical considerations when adapting for commercial use.

---

*This project demonstrates end-to-end data science capabilities including data preprocessing, exploratory analysis, machine learning modeling, and business intelligence visualization for customer churn prediction and retention strategy development.*