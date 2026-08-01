import pandas as pd

train = pd.read_json("dataset/train.json", lines=True)

print("Total Records:", len(train))

print("\nTranscript:\n")
print(train.loc[0, "transcript"])

print("\nOriginal Summary:\n")
print(train.loc[0, "summary"])