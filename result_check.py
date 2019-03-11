import pandas as pd


# ---------- Use this file to check, filter, and sort prediction result ----------
result = pd.read_csv("uc_result.txt", sep="\s+", header=None)
result = result.sort_values(by=[8])
result = result.loc[result[1] == 'uc']
result = result.loc[result[3] == 1.0]
result = result.loc[result[4] is False]
result.to_csv("ia_result.csv")
print(result)
