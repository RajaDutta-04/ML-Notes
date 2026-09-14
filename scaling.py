import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
data = {
    "Age":[20,25,30,35],
    "Salary":[30000,50000,80000,100000]
}

df = pd.DataFrame(data=data)
#print(df.head())

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df);
print("Standard Scaler: \n")
print(scaled_data)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
print("MinMax Scaler:\n")
print(scaled_data)
