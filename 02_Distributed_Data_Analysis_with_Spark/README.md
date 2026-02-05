# Superstore Sales Dataset Analysis using Apache Spark

Group Project — Cloud Technologies & Big Data Frameworks

## Goal
This project explores sales performance, customer behavior, and logistics efficiency using a large transactional retail dataset.
The main objective is to understand how Apache Spark DataFrames can be used to design scalable analytical pipelines for business-oriented questions.
## Data
-	**Dataset**: Superstore Sales Dataset
-	**Source**: Public retail dataset (Kaggle)
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
•	Loaded and structured transactional retail data using Spark DataFrames
•	Designed multiple intermediate DataFrames to support analysis
•	Performed aggregations by:
o	product category and sub-category,
o	customer segment,
o	city and region,
o	time (daily / monthly patterns)
•	Analyzed shipping performance and delivery efficiency
•	Implemented customer value and loyalty segmentation logic
•	Constructed derived metrics such as a city-level “wealth index”
Each analytical question was implemented through a dedicated Spark pipeline.

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

