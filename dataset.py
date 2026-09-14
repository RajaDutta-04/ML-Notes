import pandas as pd

dataset = {
    'Name' : ['Raja','Sanai','Rupak','Arnesh'],
    'Age' : [22,21,23,None],
    'Salary' : [50000,55000,None,80000]
}

df = pd.DataFrame(data=dataset)

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
print(df.head())