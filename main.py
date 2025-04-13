import pandas as pd

# Load the KSAT dataset
df = pd.read_csv("USKSAT_OpenRefined_Cleaned.csv")

# Show shape
print("📊 Dataset shape (rows, columns):", df.shape)

# Show basic info
print("\n🔍 Dataset info:")
print(df.info())

# Show a few rows
print("\n🧠 First few rows:")
print(df.head())
