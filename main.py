# Step 1: Import the essential libraries
import pandas as pd

# Step 2: Load the dataset
df = pd.read_csv("USKSAT_OpenRefined_Cleaned.csv")

# Step 3: View dataset shape (rows, columns)
print("Dataset shape:", df.shape)

# Step 4: View column names and datatypes
print("\nDataset info:")
print(df.info())

# Step 5: Preview first few rows
print("\nFirst 5 rows:")
print(df.head())
