import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split,cross_validate, GridSearchCV
from sklearn.impute import KNNImputer
from statsmodels.stats.proportion import proportions_ztest


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

df_ = pd.read_csv("/Users/emilcollu/PycharmProjects/TitanicCompetition/datasets/train.csv")
df = df_.copy()

df.head()
df.shape
df.isnull().sum()
df.info()

####################### DATA CLEANING ##########################

#Converting column names to uppercase
df.columns = df.columns.map(lambda x: x.upper())

#Checking to see if the target variable is balanced or not
survival_ratios = df["SURVIVED"].value_counts(normalize=True)

sns.barplot(x=survival_ratios.index,
            y=survival_ratios.values,
            palette=["red", "green"])
plt.xlabel("Survived")
plt.ylabel("Proportion")
plt.title("Survival Proportion in the Titanic Dataset")
plt.show()

#Defining outlier handling functions
def outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    Q1 = dataframe[variable].quantile(q1)
    Q3 = dataframe[variable].quantile(q3)
    IQR = Q3 - Q1
    upper_limit = q3 + 1.5 * IQR
    lower_limit = q1 - 1.5 * IQR

    return upper_limit, lower_limit

def check_outlier(dataframe, variable):
    upper_limit, lower_limit = outlier_thresholds(dataframe, variable)
    outliers = dataframe[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)]
    print(f"{variable}: {len(outliers)} outliers found.")

def replace_outliers(dataframe, variable):
    upper_limit, lower_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] > upper_limit, variable] = upper_limit
    dataframe.loc[dataframe[variable] < lower_limit, variable] = lower_limit

#Defining column identifier function
def grab_cols(dataframe, cat_th = 10, car_th = 20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() > car_th]
    cat_cols = [col for col in cat_cols + num_but_cat if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

#Grabbing cols
cat_cols, num_cols, cat_but_car = grab_cols(df)
num_cols.remove("PASSENGERID")
cat_cols = [col for col in cat_cols if col not in ["SIBSP", "PARCH"]]
num_cols.append("SIBSP")
num_cols.append("PARCH")

#Checking for outliers in numerical columns
for col in num_cols:
    check_outlier(df, col)

#Handling outliers found
for col in num_cols:
    replace_outliers(df, col)

#Checking again to make sure they are gone
for col in num_cols:
    check_outlier(df, col)

#Selecting numerical null columns
na_num_cols = [col for col in df.columns if df[col].dtypes != "O" and df[col].isnull().sum() > 0]

#Defining the function to fill null values in numerical columns with KNN imputer
def knn_imputation(dataframe, cols):
    scaler = MinMaxScaler()
    dataframe_scaled = pd.DataFrame(scaler.fit_transform(dataframe[cols]), columns=cols)

    #Applying the KNN imputer
    imputer = KNNImputer(n_neighbors=5)
    dataframe_imputed = pd.DataFrame(imputer.fit_transform(dataframe_scaled), columns=cols)

    #Reversing the scaling
    dataframe[cols] = scaler.inverse_transform(dataframe_imputed)

    return dataframe

#Filling null values
knn_imputation(df, na_num_cols)

#Checking to see if null values are gone
df[na_num_cols].isnull().sum()

#Filling small amount of null values which appeared in "EMBARKED" column with mode imputation
df["EMBARKED"] = df["EMBARKED"].fillna(df["EMBARKED"].mode()[0])
df.isnull().sum()

#Keeping the null values in "CABIN" column for feature extraction later on

#Defining a function to see target column's mean in each categorical column
def cat_summary(dataframe, col):
    print(pd.DataFrame({"Count": dataframe[col].value_counts(),
                        "Ratio": dataframe[col].value_counts() * 100 / len(dataframe),
                        "Survived Rate": dataframe.groupby(col)["SURVIVED"].mean()}))
    print("##################################################")

#Checking the summary
for col in cat_cols:
    cat_summary(df, col)

#Defining a function to see numerical column's distribution
def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles))

#Checking the summary
for col in num_cols:
    num_summary(df, col)

########################### DATA PREPROCESSING ######################

#Feature Extraction

#NEW_IS_ALONE
df["NEW_IS_ALONE"] = np.where((df["SIBSP"] + df["PARCH"] == 0), 1, 0)

#Data Viz of Survival Rate
sns.barplot(df,x="NEW_IS_ALONE", y="SURVIVED", hue="NEW_IS_ALONE", legend=False)
plt.xticks([0, 1], ["Not Alone", "Alone"])
plt.xlabel("Alone or Not")
plt.ylabel("Survival Rate")
plt.title("Effect of Being Alone on Survival Rate")
plt.show()

#Applying proportions_ztest to see if there is a statistically significant difference in survival rate between the two categories
#If p-value < 0.05 I will keep the feature created, otherwise I will drop it since there is no significant difference between the two categories

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == 1, "SURVIVED"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == 0, "SURVIVED"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == 1, "SURVIVED"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == 0, "SURVIVED"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#NEW_HAS_CABIN
df["NEW_HAS_CABIN"] = df["CABIN"].notnull().astype("int")

#Data Viz of Survival Rate
sns.barplot(df,x="NEW_HAS_CABIN", y="SURVIVED", hue="NEW_HAS_CABIN", legend=False)
plt.xticks([0, 1], ["No Cabin", "Cabin"])
plt.xlabel("No Cabin or Cabin")
plt.ylabel("Survival Rate")
plt.title("Effect of Having a Cabin on Survival Rate")
plt.show()

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_HAS_CABIN"] == 1, "SURVIVED"].sum(),
                                             df.loc[df["NEW_HAS_CABIN"] == 0, "SURVIVED"].sum()],

                                      nobs=[df.loc[df["NEW_HAS_CABIN"] == 1, "SURVIVED"].shape[0],
                                            df.loc[df["NEW_HAS_CABIN"] == 0, "SURVIVED"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#NEW_IS_DR
df["NEW_IS_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

#Data Viz of Survival Rate
sns.barplot(df,x="NEW_IS_DR", y="SURVIVED", hue="NEW_IS_DR", legend=False)
plt.xticks([0, 1], ["Not a Doctor", "Doctor"])
plt.xlabel("Doctor or Not")
plt.ylabel("Survival Rate")
plt.title("Effect of Being a Doctor on Survival Rate")
plt.show()

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_DR"] == 1, "SURVIVED"].sum(),
                                             df.loc[df["NEW_IS_DR"] == 0, "SURVIVED"].sum()],

                                      nobs=[df.loc[df["NEW_IS_DR"] == 1, "SURVIVED"].shape[0],
                                            df.loc[df["NEW_IS_DR"] == 0, "SURVIVED"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

df.drop("NEW_IS_DR", axis = 1, inplace=True)

#NEW_IS_POOR
df["NEW_IS_POOR"] = np.where((df["PCLASS"] == 3) & (df["EMBARKED"] == "Q"), 1, 0)

#Data Viz of Survival Rate
sns.barplot(df,x="NEW_IS_POOR", y="SURVIVED", hue="NEW_IS_POOR", legend=False)
plt.xticks([0, 1], ["Not Poor", "Poor"])
plt.xlabel("Poor or Not")
plt.ylabel("Survival Rate")
plt.title("Effect of Being Poor on Survival Rate")
plt.show()

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_POOR"] == 1, "SURVIVED"].sum(),
                                             df.loc[df["NEW_IS_POOR"] == 0, "SURVIVED"].sum()],

                                      nobs=[df.loc[df["NEW_IS_POOR"] == 1, "SURVIVED"].shape[0],
                                            df.loc[df["NEW_IS_POOR"] == 0, "SURVIVED"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

df.drop("NEW_IS_POOR", axis = 1, inplace=True)

#NEW_SEX_GENDER
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_AGE'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_AGE'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_AGE'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_AGE'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_AGE'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_AGE'] = 'seniorfemale'

#Data Viz of Survival Rate
custom_colors = {
    'youngmale': 'lightblue',
    'maturemale': 'mediumblue',
    'seniormale': 'darkblue',
    'youngfemale': 'lightpink',
    'maturefemale': 'hotpink',
    'seniorfemale': 'deeppink'
}
# Data Viz of Survival Rate with custom colors
sns.barplot(data=df, x="NEW_SEX_AGE", y="SURVIVED", palette=custom_colors)
plt.xlabel("Age_Gender Category")
plt.xticks(rotation=45)
plt.ylabel("Survival Rate")
plt.title("Effect of Being a Certain Age_Gender Category on Survival Rate")
plt.show()
#It's cleary seen that female passengers have way higher survival rates compared to male passengers

#NEW_FAMILY_SIZE
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

#NEW_AGE_PCLASS
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

#NEW_AGE_FARE
#Scaling FARE values between 1-5 to avoid extreme cases
scaler = MinMaxScaler(feature_range=(1, 5))
df["FARE_SCALED"] = scaler.fit_transform(df[["FARE"]])

#NEW_AGE_FARE
df["NEW_AGE_FARE"] = df["AGE"] * df["FARE_SCALED"]

#Dropping unnecessary columns after being used for feature extraction
df.drop(["CABIN", "NAME", "TICKET", "FARE_SCALED"], axis = 1, inplace=True)

#Grabbing the cols once again
cat_cols, num_cols, cat_but_car = grab_cols(df)
num_cols.remove("PASSENGERID")
cat_cols = [col for col in cat_cols if col not in ["NEW_FAMILY_SIZE", "SIBSP", "PARCH"]]
num_cols.append("SIBSP")
num_cols.append("PARCH")
num_cols.append("NEW_FAMILY_SIZE")

#Encoding

#Defining encoder functions
def label_encoder(dataframe, binary_col):
    encoder = LabelEncoder()
    dataframe[binary_col] = encoder.fit_transform(dataframe[binary_col])

    return dataframe

def one_hot_encoder(dataframe, non_ordinal_cat_col, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns=non_ordinal_cat_cols, drop_first=drop_first)

    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() == 2]

non_ordinal_cat_cols = [col for col in df.columns if df[col].dtype == "O" and 10>= df[col].nunique() > 2]

#Applying encoders
for col in binary_cols:
    label_encoder(df, col)

df = one_hot_encoder(df, non_ordinal_cat_cols)

#Converting True/False values to 1 and 0 for best practice
df = df.apply(lambda x: x.astype(int) if x.dtype == "bool" else x)

#Scaling
scaler = RobustScaler()
#Scaling numerical columns to prepare the data for modeling
df[num_cols] = scaler.fit_transform(df[num_cols])

############################# MODELING ###########################

y = df["SURVIVED"]
X = df.drop(["SURVIVED", "PASSENGERID"], axis = 1)

#Model Validation: Holdout
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier().fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print(classification_report(y_test, y_pred))
#Accuracy: 0.82
#Precision: 0.79
#Recall: 0.77
#F1-Score: 0.78

#Further Model Validation: 5-Fold Cross Validation
cv_results = cross_validate(rf_model,
                            X_train,
                            y_train,
                            cv = 5,
                            scoring =["accuracy", "precision", "recall", "f1"])

cv_results["test_accuracy"].mean()
#Accuracy: 0.80
cv_results["test_precision"].mean()
#Precision: 0.75
cv_results["test_recall"].mean()
#Recall: 0.72
cv_results["test_f1"].mean()
#F1-score: 0.73

#RandomForestClassifier model optimization with hyperparameter tuning to increase it's scores.
# rf_model.get_params()
#
# #Finding best parameters
# rf_params = {"n_estimators": range(100,500)}
#
# rf_gs_best = GridSearchCV(rf_model,
#                           rf_params,
#                           cv = 5,
#                           n_jobs= -1,
#                           verbose=1).fit(X_train, y_train)
#
# rf_gs_best.best_params_
#
# #Re-building the model with best_params_
# rf_final_model = rf_model.set_params(**rf_gs_best.best_params_).fit(X_train, y_train)
#
# y_pred = rf_final_model.predict(X_test)
# print(classification_report(y_test,y_pred))
# #Accuracy: 0.82
# #Precision: 0.79
# #Recall: 0.76
# #F1-Score: 0.77
#
# cv_results = cross_validate(rf_final_model,
#                             X_train,
#                             y_train,
#                             cv=5,
#                             scoring=["accuracy", "precision", "recall", "f1"])
#
# cv_results["test_accuracy"].mean()
# #Accuracy: 0.81
# cv_results["test_precision"].mean()
# #Precision: 0.75
# cv_results["test_recall"].mean()
# #Recall: 0.73
# cv_results["test_f1"].mean()
# #F1-score: 0.74
#
# #We see an increase in scores with cross validation after hyperparameter tuning

#Also trying KNN Model too
knn_model = KNeighborsClassifier().fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
print(classification_report(y_test, y_pred))
#Accuracy: 0.82
#Precision: 0.83
#Recall: 0.77
#F1-Score: 0.80

#Applying 5-Fold cross validation
cv_results = cross_validate(knn_model,
                            X_train,
                            y_train,
                            cv = 5,
                            scoring = ["accuracy", "precision", "recall", "f1"])

cv_results["test_accuracy"].mean()
#Accuracy: 0.77
cv_results["test_precision"].mean()
#Precision: 0.72
cv_results["test_recall"].mean()
#Recall: 0.64
cv_results["test_f1"].mean()
#F1-score: 0.68

#We can see here that the first split we picked in the Holdout of KNN-Model was probably biased since
#the success scores decreased drastically after 5-fold cross validation.

#Hyperparameter tuning for knn_model optimization
knn_model.get_params()

#finding best parameter for neighbor number between 2-50
knn_params = {"n_neighbors": range(2,50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv = 10,
                           n_jobs=-1,
                           verbose=1).fit(X, y)
knn_gs_best.best_params_

#Re-building the model with best_params_
knn_final_model = knn_model.set_params(**knn_gs_best.best_params_).fit(X_train, y_train)

y_pred = knn_final_model.predict(X_test)
print(classification_report(y_test, y_pred))

cv_results = cross_validate(knn_final_model,
                            X_train,
                            y_train,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1"])

cv_results["test_accuracy"].mean()
#Accuracy: 0.78
cv_results["test_precision"].mean()
#Precision: 0.73
cv_results["test_recall"].mean()
#Recall: 0.66
cv_results["test_f1"].mean()
#F1-score: 0.69

#XGBoost model
xgb_model = xgb.XGBClassifier(
    n_estimators=100,      # Number of boosting rounds
    learning_rate=0.05,    # Step size shrinkage
    max_depth=6,           # Depth of each tree
    objective='binary:logistic',  # Binary classification problem
    eval_metric='logloss',     # Logarithmic loss function
    random_state=42
).fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))
#Accuracy: 0.85
#Precision: 0.86
#Recall: 0.76
#F1-Score: 0.81

cv_results = cross_validate(xgb_model,
                            X_train,
                            y_train,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1"])

cv_results["test_accuracy"].mean()
#Accuracy: 0.82
cv_results["test_precision"].mean()
#Precision: 0.79
cv_results["test_recall"].mean()
#Recall: 0.70
cv_results["test_f1"].mean()
#F1-score: 0.74

#Hyperparameter tuning
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_gs_best = GridSearchCV(xgb_model,
                           xgb_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X_train, y_train)


xgb_gs_best.best_params_

#Re-building XGB model
xgb_final_model = xgb_model.set_params(**xgb_gs_best.best_params_).fit(X_train, y_train)

y_pred = xgb_final_model.predict(X_test)
print(classification_report(y_test, y_pred))
#Accuracy: 0.84
#Precision: 0.84
#Recall: 0.76
#F1-Score: 0.79

cv_results = cross_validate(xgb_final_model,
                            X_train,
                            y_train,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1"])

cv_results["test_accuracy"].mean()
#Accuracy: 0.84
cv_results["test_precision"].mean()
#Precision: 0.83
cv_results["test_recall"].mean()
#Recall: 0.73
cv_results["test_f1"].mean()
#F1-score: 0.78

#Overall XGBoost Classifier model gave the best results so I will use that for the competition.


#Reading the given test file by Kaggle
df_test = pd.read_csv("/Users/emilcollu/PycharmProjects/TitanicCompetition/datasets/test.csv")

#Applying same preprocessing actions to this dataset also

df_test.columns = df_test.columns.map(lambda x: x.upper())

cat_cols, num_cols, cat_but_car = grab_cols(df_test)
num_cols.remove("PASSENGERID")
cat_cols = [col for col in cat_cols if col not in ["SIBSP", "PARCH"]]
num_cols.append("SIBSP")
num_cols.append("PARCH")


for col in num_cols:
    check_outlier(df_test, col)

for col in num_cols:
    replace_outliers(df_test, col)

for col in num_cols:
    check_outlier(df_test, col)

na_num_cols = [col for col in df_test.columns if df_test[col].dtypes != "O" and df_test[col].isnull().sum() > 0]

knn_imputation(df_test, na_num_cols)

#NEW_IS_ALONE
df_test["NEW_IS_ALONE"] = np.where((df_test["SIBSP"] + df_test["PARCH"] == 0), 1, 0)

#NEW_HAS_CABIN
df_test["NEW_HAS_CABIN"] = df_test["CABIN"].notnull().astype("int")

#NEW_SEX_AGE
df_test.loc[(df_test['SEX'] == 'male') & (df_test['AGE'] <= 21), 'NEW_SEX_AGE'] = 'youngmale'
df_test.loc[(df_test['SEX'] == 'male') & (df_test['AGE'] > 21) & (df_test['AGE'] < 50), 'NEW_SEX_AGE'] = 'maturemale'
df_test.loc[(df_test['SEX'] == 'male') & (df_test['AGE'] >= 50), 'NEW_SEX_AGE'] = 'seniormale'
df_test.loc[(df_test['SEX'] == 'female') & (df_test['AGE'] <= 21), 'NEW_SEX_AGE'] = 'youngfemale'
df_test.loc[(df_test['SEX'] == 'female') & (df_test['AGE'] > 21) & (df_test['AGE'] < 50), 'NEW_SEX_AGE'] = 'maturefemale'
df_test.loc[(df_test['SEX'] == 'female') & (df_test['AGE'] >= 50), 'NEW_SEX_AGE'] = 'seniorfemale'

#NEW_FAMILY_SIZE
df_test["NEW_FAMILY_SIZE"] = df_test["SIBSP"] + df_test["PARCH"] + 1

#NEW_AGE_PCLASS
df_test["NEW_AGE_PCLASS"] = df_test["AGE"] * df_test["PCLASS"]

#NEW_AGE_FARE
#Scaling FARE values between 1-5 to avoid extreme cases
scaler = MinMaxScaler(feature_range=(1, 5))
df_test["FARE_SCALED"] = scaler.fit_transform(df_test[["FARE"]])

#NEW_AGE_FARE
df_test["NEW_AGE_FARE"] = df_test["AGE"] * df_test["FARE_SCALED"]

#Dropping unnecessary columns after being used for feature extraction
df_test.drop(["CABIN", "NAME", "TICKET", "FARE_SCALED"], axis = 1, inplace=True)

#Grabbing the cols once again
cat_cols, num_cols, cat_but_car = grab_cols(df_test)
num_cols.remove("PASSENGERID")
cat_cols = [col for col in cat_cols if col not in ["NEW_FAMILY_SIZE", "SIBSP", "PARCH"]]
num_cols.append("SIBSP")
num_cols.append("PARCH")
num_cols.append("NEW_FAMILY_SIZE")

binary_cols = [col for col in df_test.columns if df_test[col].dtype == "O" and df_test[col].nunique() == 2]

non_ordinal_cat_cols = [col for col in df_test.columns if df_test[col].dtype == "O" and 10>= df_test[col].nunique() > 2]

#Applying encoders
for col in binary_cols:
    label_encoder(df_test, col)

df_test = one_hot_encoder(df_test, non_ordinal_cat_cols)

#Converting True/False values to 1 and 0 for best practice
df_test = df_test.apply(lambda x: x.astype(int) if x.dtype == "bool" else x)

#Scaling
scaler = RobustScaler()
#Scaling numerical columns to prepare the data for modeling
df_test[num_cols] = scaler.fit_transform(df_test[num_cols])

#Testing results
X_test_final = df_test.drop(["PASSENGERID"], axis=1)

y_test_pred = xgb_final_model.predict(X_test_final)

submission = pd.DataFrame({
    'PassengerId': df_test['PASSENGERID'],
    'Survived': y_test_pred
})

submission.to_csv("submission.csv", index=False)


#Conclusion: I predicted the SURVIVED target variable with both RandomForestClassifier and XGBoost Classifier models separately.
#Even though XGBoost model looked more successful with the result scores here, when I submitted both predictions;
#RandomForestClassifier model with an accuracy of 0.77033
#XGBoost Classifier model with an accuracy of 0.76076

