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
import pyspark
from pyspark.sql import SparkSession

# Initialize SparkSession
spark = pyspark.sql.SparkSession \
   .builder \
   .appName("Python Spark SQL basic example") \
   .config('spark.driver.extraClassPath', "/Users/gautu/Documents/Projects/LinkedInLearning/postgresql-42.7.4.jar") \
   .getOrCreate()


# Load the CSV file into a PySpark DataFrame
# csv_file_path = "./data/cleveland.csv"
# df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Show DataFrame schema (optional)
# df.printSchema()

# PostgreSQL connection properties
db_properties = {
    "user": "postgres",
    "password": "admin",
    "driver": "org.postgresql.Driver"
}

# PostgreSQL JDBC URL
jdbc_url = "jdbc:postgresql://localhost:5432/clevelandData"

# Write the DataFrame to PostgreSQL
data.write.jdbc(
    url=jdbc_url,
    table="clevelandData",  # Replace with your table name
    mode="append",           # Use "overwrite" to replace existing data
    properties=db_properties
)

print("Data successfully loaded into PostgreSQL!")
