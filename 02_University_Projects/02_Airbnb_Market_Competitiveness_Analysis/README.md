# Airbnb Market Competitiveness Analysis using Statistical Modeling

Group Project — Fundamentals of Statistics

## Goal
This project analyzes the determinants of listing competitiveness in the Airbnb marketplace using statistical modeling techniques.
The objective is to understand which listing characteristics, pricing strategies, location factors, and host attributes influence listing demand. Review activity is used as a proxy for demand, and count-based regression models are applied to study how different variables affect expected review intensity.
The project also demonstrates how model diagnostics and statistical reasoning guide the selection of appropriate models for real-world data.
## Data
- **Dataset**: Airbnb Boston Listings Dataset
- **Source**: Public dataset from [Kaggle](https://www.kaggle.com/datasets/airbnb/boston)
- **Type**: Listing-level marketplace data
- **Scope**: Listing attributes, pricing, host characteristics, location, and review activity
The dataset contains detailed information on Airbnb listings in Boston, including pricing, property characteristics, host information, and review counts. These variables allow the analysis of factors influencing listing competitiveness within the platform.
## Tools & Methods
- **R** for statistical analysis
- **Generalized Linear Models (GLM)** for count data modeling
- Poisson regression for baseline count modeling
- Negative Binomial regression to address overdispersion
- Cluster-robust standard errors to account for host-level correlation
## Model diagnostics including:
- Pearson dispersion tests
- Likelihood Ratio Tests (LRT)
- AIC model comparison

The analysis focuses on statistical modeling and inference rather than machine learning prediction.

---

## What Was Done
The project follows a structured analytical workflow combining data preparation, model development, and model diagnostics.
1. Data Preparation
   - Loaded and cleaned the Airbnb listings dataset.
   - Selected relevant variables related to price, listing attributes, booking rules, and host characteristics.
   - Constructed exposure variables representing listing activity duration.
   - Filtered missing or inconsistent observations to produce the final analysis dataset.
2. Exploratory Data Analysis
   - Examined distributions of key variables such as price, review counts, and listing attributes.
   - Identified skewness and count-based characteristics of review data.
   - Explored geographic and neighborhood-level patterns.
3. Baseline Modeling (Linear Regression)
   - Initial models were tested using multiple linear regression.
   - Residual diagnostics revealed violations of key assumptions such as heteroscedasticity.
   - The count nature of the dependent variable motivated the transition to GLM models.
4. Poisson Regression
   - Implemented Poisson regression models to analyze review counts.
   - Compared full (“Kitchen Sink”) and simplified model specifications.
   - Conducted model diagnostics including AIC comparison and Likelihood Ratio Tests.
   - Pearson dispersion tests revealed severe overdispersion in the Poisson model.
5. Negative Binomial Regression
   - Implemented Negative Binomial regression to account for overdispersion.
   - Verified improved model fit through dispersion statistics and AIC comparison.
   = Interpreted Incidence Rate Ratios (IRR) to understand economic effects of predictors.
6. Model Robustness Checks
   - Evaluated coefficient stability across model specifications.
   - Applied cluster-robust standard errors at the host level to account for multiple listings owned by the same host.
   - Confirmed that key results remained statistically significant.

---

## Key Findings
- Pricing strategy strongly affects listing competitiveness, with higher prices associated with lower review intensity.
- Location matters significantly, with listings farther from the city center receiving fewer reviews.
- Booking convenience, particularly instant booking, increases expected review rates.
- Superhost status acts as a strong trust signal and significantly improves listing competitiveness.
- Some reputation metrics, such as ratings, have limited marginal impact because most listings already maintain very high scores.
- The marketplace appears to operate in a mature, quality-filtered environment, where differentiation occurs beyond basic quality standards.

---

## Notes
- This is a completed academic group project.
- Reviews are used as a proxy for demand, which may not perfectly reflect actual bookings or revenue.
- The analysis focuses on statistical relationships rather than causal inference.

---

## Contributors (Group 9)
- Arya Megananda
- Rihab Junaid Basheer Ahmed
- Pulkit Pratap Singh
- Ahmed Riadh Khezami

can you make the formating the same with what i gave you
