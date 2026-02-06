# Superstore Sales Dataset Analysis using Apache Spark

Group Project — Cloud Technologies & Big Data Frameworks

## Goal
This project explores sales performance, customer behavior, and logistics efficiency using a large transactional retail dataset.
The main objective is to understand how Apache Spark DataFrames can be used to design scalable analytical pipelines for business-oriented questions.
## Data
-	**Dataset**: Superstore Sales Dataset
-	**Source**: Public retail dataset [Kaggle](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting?resource=download)
-	**Type**: Transaction-level sales data
-	**Scope**: Orders, customers, products, shipping, geography, and time
The dataset spans multiple years and regions, making it suitable for aggregation-heavy and segmentation-based analysis.
## Tools & Methods
- **Apache Spark (PySpark)** for distributed data processing  
- Spark DataFrames and transformations  
- **pandas** for result inspection and lightweight visualization  
- **matplotlib** for exploratory plots 
The project focuses on distributed data processing concepts, not predictive modeling.

---

## What Was Done
The project was carried out through a structured analytical workflow using Apache Spark DataFrames:
1. Data Preparation
   - Loaded the Superstore transactional dataset into Spark.
   - Inspected schema and data types.
   - Performed basic cleaning and formatting to ensure consistency across fields (dates, categories, numerical values).
2. DataFrame Design
   - Constructed base transactional DataFrames as the foundation of the analysis.
   - Created intermediate DataFrames to support specific analytical questions.
   - Designed joins and aggregations to combine sales, customer, product, and shipping information.
3. Sales and Product Analysis
   - Aggregated sales and revenue by category and sub-category.
   - Identified top-performing products across different cities.
   - Analyzed revenue contribution and concentration patterns.
4. Customer Segmentation
   - Segmented customers by purchasing behavior and total spending.
   - Identified high-value (“Gold-tier”) customers.
   - Compared customer segments across time and geography.
5. Logistics and Shipping Performance
   - Analyzed delivery delays and shipping efficiency.
   - Computed failure rates for promised shipping times.
   - Compared logistics performance across cities and regions.
6. Geographic and Temporal Analysis
   - Examined sales patterns by city and region.
   - Analyzed time-based trends (monthly and daily behavior).
   - Constructed a city-level “wealth index” based on customer value concentration.
Each step was implemented using Spark DataFrame transformations (filter, groupBy, agg, joins), emphasizing clarity and analytical logic over optimization.

---

## Key Observations
- Sales and revenue concentration varies significantly across cities and categories
- A small subset of customers contributes disproportionately to total revenue
- Shipping delays and failures are unevenly distributed across regions
- Segmentation logic can reveal meaningful customer and geographic patterns even without predictive models
These observations highlight the value of structured analytical design in distributed environments.

---

## Notes
- This is a completed academic group project.
- The emphasis is on analytical reasoning and Spark DataFrame design, rather than optimization or production deployment.
- The notebook and presentation reflect the final submitted version.

---

## Contributors (Group 6)
- Arya Megananda
- Alessandro Micagni
- Khezami Ahmed
- Simão Frazão

