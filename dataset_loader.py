import pandas as pd

train = pd.read_json("dataset/train.json", lines=True)

print("Dataset Loaded Successfully!")
print("Total Records:", len(train))

# Display first 5 records
print(train.head())