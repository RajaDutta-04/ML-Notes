import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = {
    "Age":[20,25,30,35],
    "Experience":[0,2,4,6],
    "Salary":[30000,50000,80000,100000]
}

df = pd.DataFrame(data)

#First we split the data
X = df.drop(columns=["Salary"])
Y = df[["Salary"]]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


#Second we scale the inputs ... not outputs

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Third we fit it to model
model = LinearRegression()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
print("Actual value: ",Y_test)
print("Predicted value: ",y_pred)
print("Error: ",y_pred-Y_test)