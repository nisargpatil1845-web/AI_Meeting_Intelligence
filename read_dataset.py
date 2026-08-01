import pandas as pd

# Read JSON files
train = pd.read_json("dataset/train.json", lines=True)
validation = pd.read_json("dataset/validation.json", lines=True)
test = pd.read_json("dataset/test.json", lines=True)

print("Train Records:", len(train))
print("Validation Records:", len(validation))
print("Test Records:", len(test))

print(train.head())