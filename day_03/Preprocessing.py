from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import pandas as pd

a=np.array([[34,678,0.4],[45,786,0.6],[49,893,0.9],[58,700,0.1]])
df=pd.DataFrame(a,columns=["X1","X2","X3"])
print(df["X1"].mean())
print(df["X1"].std())
print((df["X3"]-df["X3"].mean())/df["X3"].std())

scaler=StandardScaler().set_output(transform="pandas")
# fitting data
print(scaler.fit(df))

df_scl=scaler.transform(df)
print(df_scl)
scaler=MinMaxScaler().set_output(transform="pandas")
scaler.fit(df)
df_scl=scaler.transform(df)
print(df_scl)
a=np.array([[34,678,0.4],[45,678,0.6],[49,678,0.9],[58,678,0.1]])
df=pd.DataFrame(a,columns=["X1","X2","X3"])
scaler=MinMaxScaler().set_output(transform="pandas")
print(scaler.fit_transform(df))

