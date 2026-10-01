import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import ( OneHotEncoder,OrdinalEncoder,LabelEncoder,StandardScaler,MinMaxScaler)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,classification_report)

print("\n================ TASK 1 ================")
data = {
    "Age": [25, 30, None, 28, 35, 42, None, 26, 31, 45, 29, 38],
   "Salary": [ 30000, 50000, 45000, None, 80000, 120000, 35000, 40000, 60000, None, 55000, 90000],
    "City": [ "Chennai", "Bangalore", "Chennai", "Mumbai", None, "Chennai", "Delhi", "Mumbai", None, "Bangalore", "Chennai", "Delhi"],
    "Education": [ "Bachelor", "Master", "Bachelor", "PhD", "Master", "PhD", "Bachelor", "Master","Bachelor", "PhD", "Master", "Bachelor"],
   "Experience": [2, 5, 4, 3, 10, 15,2, None, 6, 18, 5, 12],
    "Purchase_Amount": [ 1500, 3000, 2500, 4000, 7000, 10000, None, 2200, 3500, 12000, 3200, 6500],
    "Joining_Date": ["2024-01-10","2023-06-15", "2024-03-20","2022-11-05","2020-01-15","2019-05-22","2025-02-11","2024-08-18","2023-09-25","2018-04-12","2024-12-01","2021-07-19"],
   "Purchased": ["Yes", "No", "Yes", "No","Yes", "Yes", "No", "Yes","No", "Yes", "No", "Yes"]
}
df = pd.DataFrame(data)
# Save as CSV
df.to_csv("customer_data.csv", index=False)
print("\nCSV file created successfully!")
# Load CSV using Pandas
df = pd.read_csv("customer_data.csv")
print("\nFirst 5 rows:")
print(df.head())
print("\nData information:")
print(df.info())
print("\nStatistical description:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())
numeric_features = ["Age","Salary","Experience","Purchase_Amount"]
categorical_features = [ "City","Education"]
target = "Purchased"
print("\nNumerical columns:")
print(numeric_features)
print("\nCategorical columns:")
print(categorical_features)
print("\nTarget column:")
print(target)
print("\n================ TASK 2 ================")
numeric_imputer = SimpleImputer(strategy="median")
df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])
print("\nAfter median imputation:")
print(df[numeric_features])
categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)
df[categorical_features] = categorical_imputer.fit_transform(
    df[categorical_features]
)
print("\nAfter categorical imputation:")
print(df[categorical_features])
print("\nMissing values after imputation:")
print(df.isnull().sum())
print("\n---------- KNN IMPUTER BONUS ----------")
original_df = pd.read_csv("customer_data.csv")
knn_imputer = KNNImputer(n_neighbors=2)
knn_result = knn_imputer.fit_transform(
    original_df[numeric_features]
)
knn_df = pd.DataFrame(
    knn_result,
    columns=numeric_features
)
print("\nKNN Imputed Data:")
print(knn_df)

print("\n================ TASK 3 ================")
city_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)
city_encoded = city_encoder.fit_transform(
    df[["City"]]
)
city_columns = city_encoder.get_feature_names_out(
    ["City"]
)
city_encoded_df = pd.DataFrame(
    city_encoded,
    columns=city_columns
)
print("\nCity One-Hot Encoding:")
print(city_encoded_df)
education_encoder = OrdinalEncoder(
    categories=[ ["Bachelor", "Master", "PhD"] ]
)
education_encoded = education_encoder.fit_transform(
    df[["Education"]]
)
print("\nEducation Ordinal Encoding:")
print(education_encoded)
label_encoder = LabelEncoder()
df["Purchased_Encoded"] = label_encoder.fit_transform(
    df["Purchased"]
)
print("\nTarget Label Encoding:")
print(df[["Purchased", "Purchased_Encoded"]])

df["Joining_Date"] = pd.to_datetime(
    df["Joining_Date"]
)
df["Joining_Year"] = df["Joining_Date"].dt.year
df["Joining_Month"] = df["Joining_Date"].dt.month
df["Experience_Ratio"] = (
    df["Experience"] / df["Age"]
)
def spending_category(amount):

    if amount < 3000:
        return "Low"

    elif amount <= 6000:
        return "Medium"

    else:
        return "High"
df["Spending_Category"] = df[
    "Purchase_Amount"
].apply(
    lambda x: spending_category(x)
    if pd.notna(x)
    else "Unknown"
)
print("\nFeature engineered data:")
print(
    df[
        [ "Joining_Date", "Joining_Year","Joining_Month","Experience_Ratio","Spending_Category"]]
)
standard_scaler = StandardScaler()

standard_result = standard_scaler.fit_transform(
    df[numeric_features]
)

standard_df = pd.DataFrame(
    standard_result,
    columns=numeric_features
)
print("\nStandardScaler result:")
print(standard_df)

minmax_scaler = MinMaxScaler()

minmax_result = minmax_scaler.fit_transform(
    df[numeric_features]
)

minmax_df = pd.DataFrame(
    minmax_result,
    columns=numeric_features
)
print("\nMinMaxScaler result:")
print(minmax_df)

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)
print("\nNumerical Pipeline created:")
print(numeric_pipeline)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)
print("\nCategorical Pipeline created:")
print(categorical_pipeline)


final_numeric_features = [ "Age","Salary","Experience","Purchase_Amount","Experience_Ratio"]
final_categorical_features = [
    "City",
    "Education"
]
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            final_numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            final_categorical_features
        )
    ]
)
print("\nColumnTransformer created:")
print(preprocessor)
print("\n================ TASK 9 ================")
X = df[
    final_numeric_features +
    final_categorical_features
]

y = df["Purchased_Encoded"]
model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)
print("\nComplete ML Pipeline:")
print(model_pipeline)

print("\n================ TASK 10 ================")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining data:")
print(X_train)
print("\nTesting data:")
print(X_test)
print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
model_pipeline.fit( X_train,y_train)
y_pred = model_pipeline.predict(X_test)
print("\nPredictions:")
print(y_pred)
print("\nActual values:")
print(y_test.values)
print("\n================ MODEL RESULTS ================")
accuracy = accuracy_score( y_test,y_pred)
precision = precision_score(y_test,y_pred,zero_division=0)
recall = recall_score(y_test,y_pred,zero_division=0)
f1 = f1_score(y_test,y_pred,zero_division=0)
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred,zero_division=0))
print("\n================ TASK 11 ================")
print("""
DATA LEAKAGE EXPLANATION
------------------------
Incorrect workflow:
Entire Dataset
      |
      v
Imputer / Scaler
      |
      v
Train / Test Split
Correct workflow:
Entire Dataset
      |
      v
Train / Test Split
      |
      +----------------+
      |                |
      v                v
   Training          Testing
      |
      v
Fit Imputer / Scaler
      |
      v
Transform Training
and Testing Data
      |
      v
Machine Learning Model
Our Pipeline follows the correct workflow.
The preprocessing steps are fitted only using
the training data.
This helps prevent information from the test
data leaking into the training process.
""")
print("\n================ FINAL DATA ================")
