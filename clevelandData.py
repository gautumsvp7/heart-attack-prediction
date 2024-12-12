import pandas as pd
from charset_normalizer import from_path

file_path = "data/cleveland.csv"

# Detect encoding
detected = from_path(file_path).best()
print(f"Detected encoding: {detected.encoding}")

# Use the detected encoding to read the file
data = pd.read_csv(file_path, header=None, encoding=detected.encoding)
# print(data.head())
# print(f"Number of columns detected: {data.shape[1]}")


# Define column names (modify as per actual dataset details)
data.columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

# Display the first few rows with headers
print(data.head())

data.to_csv("clevelandWithHeader.csv", index=False)


