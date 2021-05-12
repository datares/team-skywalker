#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

rawData = pd.read_csv("Kepler_Data.csv")
rawData = rawData.drop("koi_disposition", axis = 1)
rawData
# %%
le = LabelEncoder()
y = le.fit_transform(rawData['koi_pdisposition'])
X = rawData.drop("koi_pdisposition", axis = 1)
#%%
X.to_csv("cleanData.csv")
#%%
# drop error columns
dropCols = X.filter(regex="err\d$").columns
X = X.drop(dropCols,axis = 1)
# 
numericCols = rawData.select_dtypes(np.number).columns.to_list()
X = X[numericCols]
ss = StandardScaler()

# find numeric columns
X = X.fillna(0)
X[numericCols] = ss.fit_transform(X)
X = X.drop(["rowid", "kepid"], axis = 1)

# %%
RF = RandomForestClassifier()
RF.fit(X, y)

# %%
features = pd.DataFrame(RF.feature_importances_, X.columns)
# %%
features.sort_values(by=0)
# %%
# %%
