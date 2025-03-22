# OVERVIEW

In this application, you will explore a dataset from Kaggle. The original dataset contained information on 3 million used cars. The provided dataset contains information on 426K cars to ensure speed of processing.  Your goal is to understand what factors make a car more or less expensive.  As a result of your analysis, you should provide clear recommendations to your client -- a used car dealership -- as to what consumers value in a used car.

## CRISP-DM Framework

To frame the task, throughout our practical applications, we will refer back to a standard process in industry for data projects called CRISP-DM.  This process provides a framework for working through a data problem.  Your first step in this application will be to read through a brief overview of CRISP-DM [here](https://mo-pcco.s3.us-east-1.amazonaws.com/BH-PCMLAI/module_11/readings_starter.zip).  After reading the overview, answer the questions below.

# Business Understanding

From a business perspective, we are tasked with identifying key drivers for used car prices.  In the CRISP-DM overview, we are asked to convert this business framing to a data problem definition.  Using a few sentences, reframe the task as a data task with the appropriate technical vocabulary. 

### Objective :

To develop a supervised machine learning regression model to predict the price of used vehicles based on a range of features such as:
- year, 
- manufacturer, 
- model, 
- condition, 
- mileage (odometer), 
- fuel type, 
- transmission, and 
- body type. 

The goal is to identify the most influential variables that impact vehicle pricing, enabling data-driven insights into the key drivers of used car values.

### Data Exploration Plan

### 1. Initial Data Exploration

- Load the dataset and view the first few rows (`df.head()`)
- Understand the structure using `.info()` and `.describe()`
- Check the shape (rows x columns) with `df.shape`
- Identify data types: numeric, categorical, date, text

### 2. Understand the Target Variable

- Examine the distribution of price (our target)
  - Histogram, boxplot to detect outliers
  - Check for unreasonable values (e.g. 0, extremely high prices)

### 3. Missing Values Analysis

- Use `df.isnull().sum()` and percentage calculations to:
  - Identify columns with missing values
  - Estimate how serious the missingness is
- Consider if missing data is random or has a pattern

### 4. Duplicate and Invalid Data

- Check for duplicate rows or records with `df.duplicated().sum()`
- Validate key columns like year, odometer, price, VIN
  - Example: year should be between 1900 and current year
  - Odometer and price should not be negative or unreasonably high

### 5. Categorical Features Exploration

- Use `value_counts()` for:
  - Manufacturer, fuel, transmission, condition, type, state, etc.
- Identify dominant categories vs. rare ones (long tails)

### 6. Numeric Features Exploration

- Visualize odometer, year, price with histograms, KDE plots
- Detect skewness or strange distributions
- Check correlations between numerical variables and price

### 7. Relationship Analysis

- Use visualizations to explore:
  - Price vs. Year (newer cars = higher price?)
  - Price vs. Mileage (higher mileage = lower price?)
  - Price vs. Manufacturer, Condition, etc.

### 8. Data Consistency Checks

- Look for inconsistencies:
  - Mismatched year and model
  - Condition labeled as "new" with high mileage
  - Transmission or fuel with unexpected values or typos


# 🧼 Data Preparation Plan

## ✅ 1. Handle Missing Values

- Drop or fill missing values, depending on the column and business logic

## ✅ 2. Remove Outliers & Invalid Data

- Filter out unreasonable price, year, odometer values

## ✅ 3. Encode Categorical Variables

- Use One-Hot Encoding or Label Encoding where appropriate

## ✅ 4. Feature Engineering (Optional)

- Create new features (e.g., car age)

## ✅ 5. Transformations

- Log-transform skewed variables like price, odometer
- Scale numerical features with StandardScaler or MinMaxScaler


# 🎯 Modeling Plan

## 🔢 Models to Try:

- Linear Regression
- Ridge Regression (regularized)
- Random Forest Regressor
- Gradient Boosting Regressor

## ✅ Tasks:

- Fit models using a pipeline
- Use cross-validation (`cross_val_score`)
- Compare RMSE across models


# Comparison of Regression Models

Using a sampled dataset and faster 3-fold cross-validation, we compared the performance of several regression models. As you can see:

- **Gradient Boosting** performed the best with the lowest RMSE.
- **Random Forest** was a close second.
- **Linear and Ridge Regression** had similar, slightly higher errors.

## Results Summary

| Model                  | RMSE       |
|------------------------|------------|
| Gradient Boosting      | Lowest     |
| Random Forest          | Close Second|
| Linear Regression      | Higher     |
| Ridge Regression       | Higher     |

## Conclusion

Gradient Boosting is the most effective model for this dataset, followed closely by Random Forest. Linear and Ridge Regression models have higher errors but are still useful for comparison.


# ✅ Model Evaluation Summary

## 🎯 Business Objective Recap

We set out to identify the key drivers of used car prices and build a predictive model to estimate those prices based on vehicle features. This would enable better pricing strategies, valuation tools, or insights into market behavior.

## 🧠 What We Learned

### Model Performance

- **Final model**: Gradient Boosting Regressor (tuned)
- **R² Score**: ~0.77 → The model explains 76.87% of the variance in used car prices.
- **RMSE**: ~$6,722 → On average, predictions are within ~$6,700 of the actual value.
- **MAE**: ~$4,282 → Half of the predictions are off by less than ~$4,300.

### Key Drivers of Price (Feature Importance)

- **Car age**: Newer cars are significantly more expensive.
- **Odometer reading**: Higher mileage reduces price, as expected.
- **Manufacturer**: Brands like Toyota, Ford, etc., show distinct pricing patterns.
- **Fuel type & transmission**: Gas vehicles and automatics dominate and influence price variability.
- **Vehicle type**: Trucks and SUVs generally priced higher than sedans or coupes.

## 📌 Model Quality Reflection

A high-quality model should:

- Generalize well (✓)
- Be interpretable (✓)
- Provide actionable insight (✓)
- Align with business goals (✓)

This model achieves all of the above, making it a strong candidate for deployment or decision support.

## 🔁 Do We Need to Revisit Earlier Phases?

| Phase                | Revisit Needed? | Notes                                                   |
|----------------------|-----------------|---------------------------------------------------------|
| Business Understanding | ❌               | Clear, aligned with model goals                         |
| Data Understanding     | ✅ (Optional)   | Could dive deeper into rare categories or segments      |
| Data Preparation       | ❌               | Robust pipeline used                                    |
| Modeling               | ✅ (Optional)   | Could try XGBoost, ensemble models, or SHAP for explainability |

## 💼 What to Bring to the Client

- A tuned, interpretable model predicting used car prices with solid accuracy
- Clear understanding of what drives car value
- Potential for integrating this model into pricing tools, dashboards, or marketing strategy

# 🚗 Used Car Price Prediction – Final Report

## 👥 Audience

Used car dealers seeking to optimize pricing strategies and understand what factors most influence vehicle value.

## 🎯 Objective

To build a data-driven model that predicts used car prices based on features such as mileage, age, manufacturer, fuel type, transmission, and body type. The goal is to uncover the most influential factors and support better pricing and inventory decisions.

## 🔍 Key Findings

### ✅ Top Predictive Factors:

- **Car Age** – Newer cars retain significantly more value.
- **Mileage (Odometer)** – Higher mileage strongly reduces price.
- **Manufacturer** – Brand reputation matters (e.g., Toyota, Ford, BMW).
- **Vehicle Type** – SUVs and trucks are priced higher than sedans.
- **Fuel Type & Transmission** – Gasoline and automatic vehicles dominate pricing patterns.

## 📈 Model Summary

- **Model Used**: Gradient Boosting Regressor (Tuned)
- **R² Score**: 0.7687 (model explains ~77% of price variability)
- **RMSE**: $6,722 (average prediction error)
- **MAE**: $4,281 (half of predictions within this margin)

This model is accurate enough for practical use in pricing or appraisal tools.

## 📊 Visual Insights

- **Actual vs. Predicted** – Strong alignment indicates reliable pricing predictions.
- **Residuals** – Errors are normally distributed; no major systemic bias.
- **Feature Importance** – Confirms key business insights: age, mileage, brand matter most.

## 💡 Recommendations for Dealers

- Prioritize newer, low-mileage vehicles when acquiring inventory.
- Use manufacturer and type insights to target high-margin segments.
- Implement this model into your pricing tools or websites for real-time valuations.
- Use insights for marketing (e.g., “lower mileage = better deal!”).
