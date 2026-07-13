# Applied AI & ML Capstone Project

## Part 1 – Data Acquisition, Cleaning, and Exploratory Data Analysis (EDA)

### Student

**Pallavi Gupta**

### Course

Masai School × IIT Patna – Applied AI & ML Capstone Project

---

# Dataset Used

**Dataset:** Ames Housing Dataset

The Ames Housing dataset is a publicly available dataset containing detailed information about residential properties in Ames, Iowa. It consists of **2930 observations** and **82 features**, including both numerical and categorical variables.

### Why this dataset?

This dataset was selected because it satisfies all the project requirements:

* More than 500 rows
* More than 5 numerical columns
* More than 2 categorical columns
* Clearly defined target variable (`SalePrice`)
* Suitable for both regression (predicting SalePrice) and classification (by binning SalePrice into categories in later parts)

The dataset is widely used for machine learning, data preprocessing, feature engineering, and regression analysis.

---

# Objectives

The objectives of Part 1 are:

* Load and inspect the raw dataset.
* Identify missing values.
* Handle missing data appropriately.
* Detect duplicate records.
* Optimize data types.
* Analyze descriptive statistics.
* Measure skewness.
* Detect outliers using the IQR method.
* Perform exploratory data analysis through visualizations.
* Compare Pearson and Spearman correlations.
* Perform grouped aggregation.
* Save a cleaned dataset for future modeling.

---

# Data Loading

The dataset was loaded using **pandas**.

Initial inspection included:

* First five rows
* Dataset shape
* Data types
* Dataset information

Dataset Shape:

* **Rows:** 2930
* **Columns:** 82

---

# Missing Value Analysis

Missing values were calculated for every column using:

* Missing Count
* Missing Percentage

Columns exceeding a **20% missing rate** were identified:

* Pool QC
* Misc Feature
* Alley
* Fence
* Mas Vnr Type
* Fireplace Qu

For numerical columns with less than 20% missing values, missing values were filled using the **median**.

### Why Median?

The median is more robust than the mean because it is less affected by extreme values and skewed distributions. Since housing datasets often contain outliers, median imputation provides a better estimate of the central tendency.

---

# Duplicate Detection

Duplicate rows were checked using `df.duplicated().sum()`.

Results:

* Duplicate rows found: **0**
* Rows removed: **0**

Since no duplicate rows were present, the dataset size and missing value percentages remained unchanged.

---

# Data Type Optimization

To improve memory efficiency and represent the data more accurately:

* `MS SubClass` was converted from integer to **category** because it represents categories rather than numerical quantities.
* `Neighborhood` was converted from string to **category** because it contains repeated categorical values.

Memory Usage:

* Before conversion: **7,251,298 bytes**
* After conversion: **7,072,791 bytes**
* Memory Saved: **178,507 bytes**

---

# Descriptive Statistics

Summary statistics were generated for all numerical columns using `df.describe()`.

The analysis included:

* Count
* Mean
* Standard Deviation
* Minimum
* Quartiles
* Maximum

These statistics provide an overview of the numerical characteristics of the dataset.

---

# Skewness Analysis

Skewness was calculated for every numerical feature.

The most skewed feature was:

* **Misc Val**
* **Skewness = 22.000**

This indicates an extremely **positively skewed** distribution.

Most houses have a miscellaneous value of zero, while a small number of houses contain very large values, creating a long right tail.

Because of this positive skewness, the **median** is a better measure of central tendency than the mean for handling missing values.

---

# Outlier Detection (IQR)

The Interquartile Range (IQR) method was used to detect outliers.

The following columns were analyzed:

### SalePrice

* Outliers detected: **137**

### Gr Liv Area

* Outliers detected: **75**

No outliers were removed.

These observations likely represent genuine expensive or unusually large houses rather than incorrect data entries. They will be retained for future modeling, where appropriate algorithms can handle them effectively.

---

# Visualizations

The following visualizations were created:

### Line Plot

A line plot showing the variation of **SalePrice** across the dataset.

---

### Bar Chart

Average **SalePrice** grouped by **MS Zoning**.

---

### Histogram

Histogram of **Misc Val**, the most skewed numerical feature.

The histogram shows that most observations are concentrated near zero with a long right tail, confirming the calculated positive skewness.

---

### Scatter Plot

Scatter plot between:

* Gr Liv Area
* SalePrice

The scatter plot shows a clear positive relationship, indicating that houses with larger living areas generally have higher sale prices.

---

### Box Plot

Box plot of SalePrice grouped by MS Zoning.

The visualization shows noticeable differences in median sale prices and variation among zoning categories, indicating that zoning influences housing prices.

---

### Correlation Heatmap

A Pearson correlation heatmap was generated for all numerical variables.

The highest absolute correlation was observed between **Order** and **Yr Sold** (correlation ≈ **0.976**).

This strong correlation does not necessarily indicate a causal relationship. A plausible explanation is that the dataset is organized chronologically, so the order in which records appear is naturally associated with the year of sale rather than one variable causing the other.
---

# Imputation Strategy Comparison

The two most skewed numerical variables were analyzed using both the mean and median.

Because the distributions were highly skewed, the **median** was selected for imputation since it is more resistant to extreme values than the mean.

After imputation, the selected numerical columns contained no remaining missing values.

---

# Pearson vs Spearman Correlation

Both Pearson and Spearman correlation matrices were computed.

The three variable pairs with the largest absolute differences between Pearson and Spearman correlations were identified.

When Spearman correlation exceeded Pearson correlation, the relationship was interpreted as monotonic but not perfectly linear.

For feature selection in future modeling, Pearson correlation will be preferred for approximately linear relationships, while Spearman correlation will be considered for monotonic non-linear relationships.

---

# Grouped Aggregation

The dataset was grouped using **MS Zoning**, and summary statistics of **SalePrice** were calculated.

The analysis included:

* Mean
* Standard Deviation
* Count

The group with the highest mean sale price and the group with the highest variability were identified.

The ratio between the highest and lowest average sale prices suggests that zoning carries useful predictive information and may contribute to future machine learning models.

---

# Cleaned Dataset

The cleaned dataset was saved as:

`cleaned_data.csv`

This dataset will be used in Part 2 and Part 3 of the capstone project.

---

# Libraries Used

* pandas
* numpy
* matplotlib
* seaborn

---

# Conclusion

In this part, the Ames Housing dataset was successfully cleaned and explored through comprehensive exploratory data analysis. Missing values were handled appropriately, duplicate records were verified, data types were optimized, skewness and outliers were analyzed, multiple visualizations were generated, correlations were examined, and grouped statistical summaries were produced. The cleaned dataset is now ready for feature engineering and machine learning in the next phase of the project.
