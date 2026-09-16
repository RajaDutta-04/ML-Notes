import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    "Hours":[1,2,3,4,5,6],
    "Result":[0,0,0,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
Y = df["Result"]

model = LogisticRegression()

model.fit(X,Y)

a = int(input("Enter hours: "))
if(a < 0):
    print("Invalid !")
else:
    prediction = model.predict([[a]])[0]
    if(prediction == 1) :
        print("Candidate is likely to be passed!")
    else:
        print("Candidate is likely to be failed!")
