# Salary Prediction using Linear Regression and Gradient Boostong Regressor

## Problem - 
Predict salaray based on multiple features.

## Data
  ### What we have 
  - We have 2 files - Train and Test File. 
  - Train file has 100k observations with 7 features
  - 4 categorical and 2 numerical data

We have data as in below
![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/Original_data.png)

## Data Preparataion 
We ran through data processing to look for following
  - Nulls
  - Data types to see if numerical columns are marked as object
  - how many categorical and numericals columns in the dataframe

## Feature Engineering
### Hot Encoding the categorical values
Used hot encoding to convert the categorical values to numerical values as below, as the models only work on the numerical columns
![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/hont_encoding.png)

### Correlation features to Salary
Evaluated the correlation to see which featured need to be considered. 
![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/numerical_corelation.png)

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/companyid_corr.png)

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/degree.png)

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/jobtype.png)

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/major.png)

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/major.png)

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/industry.png)

From the Correlation, Company ID  doenst have impact on Salary, so will be ignored.

## Model
Evaluated 2 models - Linear Regression and Gradient Boosting Regresso

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/LR_MSE.png)

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/gbr_MSE.pngg)

## Conclussion
From MSE numbers, GBR seems to be better model.

Used GBR to predict final Salaries.

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/Predicated_sal.png)

Using GBR, evaluated the Features to see which has more impact

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/feature_evaluation.png)
