import pandas as pd

# Load the CSV
file_path = "./framingham_with_id.csv"
data = pd.read_csv(file_path)

# Quick inspection
print(data.head())
# print(data.info())
