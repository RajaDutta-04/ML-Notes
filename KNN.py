import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data = {
    'Hours':[1,2,3,4,5,6],
    'Attendance':[50,55,60,65,70,75],
    'Result':[0,0,0,1,1,1]
}

df = pd.DataFrame(data)

model = KNeighborsClassifier(n_neighbors=3)
scale = StandardScaler()

X = df[['Hours','Attendance']]
x = scale.fit_transform(X)
y = df['Result']

model.fit(x,y)

pred = model.predict([[2,85]])[0]

print(pred)