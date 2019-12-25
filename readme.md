# Project

A simple machine learning pipeline is created that will ingest the dataset and feed into 2 machine learning algorithms, returning metric outputs.

## Submission files

  1. Exploratory Data Analysis (EDA): *eda2.ipynb*
  2. Python Module: *modulescript.py*
  3. Executable bash script: *run.sh*
  4. Requirements file: *requirements.txt*

## Highlights of EDA

Plotly and cufflinks are used to create an interactive notebook in Python.

From the EDA, traffic peaks during holidays and non-holidays follow a different pattern. 
 
## Highlights of Machine Learning Pipeline
### Data Preparation
Only time 0000hrs has the name of the holiday. Inpute name of holiday into the relevant rows under the 'holiday' column. 

Drop variable **snow_1h**, as the values are uniformed throughout all data points. The weather description mentions that there was snow in the year, but this variable "snow_1h" indicates otherwise. This is likely error in data collection and the variable "snow_1h" is removed.

Create feature **hol** to store values of whether it is a holiday or not.

Create feature **dow** to represent day of week which is derived from the **date** attribute.

Create feature **typeofday** to represent weekday or weekend.

Normalise data for machine learning algorithms to work on.

### Machine Learning 
80% of data points went to Training Data and 20% to Validation Data.

Two Machine Learning algorithms:
  - Linear Regression
  - Random Forest

### Result
  - Random Forest had a lower Mean Squared Error than Linear Regression.

## Requirements
Install the dependencies.

```sh
$ pip install -r requirements.txt
```

### 
This markdown document was created using Dillinger online markdown editor.
