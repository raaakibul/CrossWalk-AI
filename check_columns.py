import pandas as pd

file = "Survey-data-on-pedestrian-crosswalk-usage-behavior-in-Dhaka.xlsx"

xls = pd.ExcelFile(file)
print("Sheet names:", xls.sheet_names)

df = pd.read_excel(file, sheet_name=0)
print("\nColumns found:")
for i, col in enumerate(df.columns):
    print(i, repr(col))

print("\nFirst 5 rows:")
print(df.head())