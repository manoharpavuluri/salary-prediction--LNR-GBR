# Salary Prediction using Linear Regression and Gradient Boosting Regressor

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
Evaluated 2 models - Linear Regression and Gradient Boosting Regressor

#### Linear Regression
Predicted VS Real Plot

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/lr_prediction_plot.png)

MSE Evaluation

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/LR_MSE.png)


#### Gradient Boosting Regressor
Predicted VS Real Plot

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/gbr_prediction_plot.png)

MSE Evaluation

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/gbr_MSE.png)

## Conclussion
Although Predicted VS Real plots looks same, from further evaluations MSE numbers, GBR seems to be better model.

Using GBR, evaluated the Features to see which has more impact

![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/feature_evaluation.png)

## And Finally the Predicted Salaries using GBR Model
![alt tex](https://github.com/manoharpavuluri/salary-prediction--LNR-GBR/blob/master/pictures/Predicated_sal.png)
