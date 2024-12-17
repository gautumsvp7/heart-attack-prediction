import pandas as pd
import pyspark
from pyspark.sql import SparkSession

# Step 1: Extract
def extract(file_path):
    print("Extracting data...")
    data = pd.read_csv(file_path, na_values=['?'], header=0)
    return data

# Step 2: Transform
def transform(data):
    print("Transforming data...")
   

    #Defining an age group column for ease of use in PowerBI
    data['age_group'] = pd.cut(data['age'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '21-40', '41-60', '61-80', '80+'])

    numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col].fillna(data[col].median(),inplace=True)
    data.to_csv("clevelandWithHeader.csv", index=False) #Saves csv

    return data

# Step 3: Load
def load(jdbc_url, table_name):
    spark = pyspark.sql.SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config('spark.driver.extraClassPath', "/Users/gautu/Documents/Projects/LinkedInLearning/postgresql-42.7.4.jar") \
    .getOrCreate()

    csv_file_path = "./clevelandWithHeader.csv"
    data = spark.read.csv(csv_file_path, header=True, inferSchema=True)



    # PostgreSQL connection properties
    db_properties = {
        "user": "postgres",
        "password": "admin",
        "driver": "org.postgresql.Driver"
    }

    # PostgreSQL JDBC URL
    # jdbc_url = "jdbc:postgresql://localhost:5432/clevelandData"

    # Write the DataFrame to PostgreSQL
    data.write.jdbc(
        url=jdbc_url,
        table= table_name,
        mode="append",  # Use "overwrite" to replace existing data
        properties=db_properties
    )

    print("Data successfully loaded into PostgreSQL!")

# Main function to run the pipeline
def run_etl(file_path, db_url, table_name):
    data = extract(file_path)
    transformed_data = transform(data)
    load(db_url, table_name)

# Define file path, database URL, and table name
file_path = "./clevelandWithHeader.csv"
db_url = "jdbc:postgresql://localhost:5432/clevelandData"
table_name = "clevelandData"

# Run the ETL pipeline
run_etl(file_path, db_url, table_name)
