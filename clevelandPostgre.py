import pyspark
from pyspark.sql import SparkSession


# Initialize SparkSession
spark = pyspark.sql.SparkSession \
   .builder \
   .appName("Python Spark SQL basic example") \
   .config('spark.driver.extraClassPath', "/Users/gautu/Documents/Projects/LinkedInLearning/postgresql-42.7.4.jar") \
   .getOrCreate()

csv_file_path = "./clevelandWithHeader.csv"
df = spark.read.csv(csv_file_path, header=True, inferSchema=True)



# PostgreSQL connection properties
db_properties = {
    "user": "postgres",
    "password": "admin",
    "driver": "org.postgresql.Driver"
}

# PostgreSQL JDBC URL
jdbc_url = "jdbc:postgresql://localhost:5432/clevelandData"

# Write the DataFrame to PostgreSQL
df.write.jdbc(
    url=jdbc_url,
    table="clevelandData",
    mode="append",  # Use "overwrite" to replace existing data
    properties=db_properties
)

print("Data successfully loaded into PostgreSQL!")