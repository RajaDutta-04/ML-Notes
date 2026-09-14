import pandas as pd
from sklearn.preprocessing import LabelEncoder 

dataset = {
    "City" : ["Kolkata","Delhi","Mumbai","Kolkata"]
}
df = pd.DataFrame(data=dataset)
encoder = LabelEncoder()
encoded_data = encoder.fit_transform(df)
print("Label Encoded Data: ",encoded_data)

df_onehot = pd.get_dummies(df,columns=["City"],dtype=int)
print("One Hot Encoded Data :\n",df_onehot)