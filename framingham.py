'''
run only once:
'''
import pandas as pd

# Load the dataset
df = pd.read_csv("framingham.csv")

# Add a synthetic primary key
df.insert(0, 'patient_id', range(1, len(df) + 1))

# Save the updated dataset
df.to_csv("framingham_with_id.csv", index=False)