#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import numpy as np


file_path = 'D:\STOCKS DATA\COCHINSHIPYARD1M.csv' 
data = pd.read_csv(file_path)
 
print(data.head())
data.columns = data.columns.str.strip()


print(data.columns)


data['Date'] = pd.to_datetime(data['Date'])

data = data.sort_values('Date')


data['Date'] = pd.to_datetime(data['Date'])

data.set_index('Date', inplace=True)
if 'Close' not in data.columns:
    print("Error: 'Close' column not found in the CSV file.")
    exit()
data.rename(columns=lambda x: x.strip(), inplace=True) 
if 'close' in data.columns:
    data.rename(columns={'close': 'Close'}, inplace=True)

data['Day'] = data.index.day
data['Month'] = data.index.month
data['Year'] = data.index.year
data['DayOfWeek'] = data.index.dayofweek

# Select features for PCA
features = ['Day', 'Month', 'Year', 'DayOfWeek']
X_features = data[features]

# Apply PCA
n_components = 2 
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(X_features)
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each component: {explained_variance}")


data['Date_num'] = data.index.map(pd.Timestamp.toordinal)

data['Close'] = data['Close'].str.replace(',', '').astype(float)
X_pca = np.hstack((principal_components, data['Date_num'].values.reshape(-1, 1)))
X = X_pca
y = data['Close'].values
model = LinearRegression()

model.fit(X, y)



future_dates = pd.date_range(data.index[-1], periods=7
                            )
future_dates_num = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)


future_days = future_dates.day
future_months = future_dates.month
future_years = future_dates.year
future_dayofweeks = future_dates.dayofweek


future_features = np.vstack((future_days, future_months, future_years, future_dayofweeks)).T
future_pca = pca.transform(future_features)


future_X_pca = np.hstack((future_pca, future_dates_num))
future_predictions = model.predict(future_X_pca)


future_data = pd.DataFrame({'Date': future_dates, 'Close': future_predictions})
future_data.set_index('Date', inplace=True)


plt.figure(figsize=(12, 6))
sns.lineplot(data=data['Close'], label='Historical Data')
sns.lineplot(data=future_data['Close'], label='Future Predictions')
plt.title('Stock Price Prediction with PCA')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()


# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import numpy as np


file_path = 'D:\STOCKS DATA\COCHINSHIPYARD3M.csv' 
data = pd.read_csv(file_path)
 
print(data.head())
data.columns = data.columns.str.strip()


print(data.columns)


data['Date'] = pd.to_datetime(data['Date'])

data = data.sort_values('Date')


data['Date'] = pd.to_datetime(data['Date'])

data.set_index('Date', inplace=True)
if 'Close' not in data.columns:
    print("Error: 'Close' column not found in the CSV file.")
    exit()
data.rename(columns=lambda x: x.strip(), inplace=True) 
if 'close' in data.columns:
    data.rename(columns={'close': 'Close'}, inplace=True)

data['Day'] = data.index.day
data['Month'] = data.index.month
data['Year'] = data.index.year
data['DayOfWeek'] = data.index.dayofweek

# Select features for PCA
features = ['Day', 'Month', 'Year', 'DayOfWeek']
X_features = data[features]

# Apply PCA
n_components = 2 
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(X_features)
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each component: {explained_variance}")


data['Date_num'] = data.index.map(pd.Timestamp.toordinal)

data['Close'] = data['Close'].str.replace(',', '').astype(float)
X_pca = np.hstack((principal_components, data['Date_num'].values.reshape(-1, 1)))
X = X_pca
y = data['Close'].values
model = LinearRegression()

model.fit(X, y)



future_dates = pd.date_range(data.index[-1], periods=7
                            )
future_dates_num = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)


future_days = future_dates.day
future_months = future_dates.month
future_years = future_dates.year
future_dayofweeks = future_dates.dayofweek


future_features = np.vstack((future_days, future_months, future_years, future_dayofweeks)).T
future_pca = pca.transform(future_features)


future_X_pca = np.hstack((future_pca, future_dates_num))
future_predictions = model.predict(future_X_pca)


future_data = pd.DataFrame({'Date': future_dates, 'Close': future_predictions})
future_data.set_index('Date', inplace=True)


plt.figure(figsize=(12, 6))
sns.lineplot(data=data['Close'], label='Historical Data')

plt.title('Stock Price Prediction with PCA')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()


# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import numpy as np


file_path = 'D:\STOCKS DATA\COCHINSHIPYARD1M.csv'
data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()
data['Date'] = pd.to_datetime(data['Date'])


data = data.sort_values('Date')


data.set_index('Date', inplace=True)


if 'Close' not in data.columns:
    print("Error: 'Close' column not found in the CSV file.")
    exit()


data.rename(columns=lambda x: x.strip(), inplace=True)
if 'close' in data.columns:
    data.rename(columns={'close': 'Close'}, inplace=True)

data['Day'] = data.index.day
data['Month'] = data.index.month
data['Year'] = data.index.year
data['DayOfWeek'] = data.index.dayofweek


features = ['Day', 'Month', 'Year', 'DayOfWeek']
X_features = data[features]

# Apply PCA
n_components = 2
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(X_features)
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each component: {explained_variance}")


data['Date_num'] = data.index.map(pd.Timestamp.toordinal)


data['Close'] = data['Close'].str.replace(',', '').astype(float)


X_pca = np.hstack((principal_components, data['Date_num'].values.reshape(-1, 1)))
X = X_pca
y = data['Close'].values


model = LinearRegression()
model.fit(X, y)


future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=7)


future_dates_num = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)


future_days = future_dates.day
future_months = future_dates.month
future_years = future_dates.year
future_dayofweeks = future_dates.dayofweek


future_features = np.vstack((future_days, future_months, future_years, future_dayofweeks)).T
future_pca = pca.transform(future_features)

 
future_X_pca = np.hstack((future_pca, future_dates_num))
future_predictions = model.predict(future_X_pca)


future_data = pd.DataFrame({'Date': future_dates, 'Close': future_predictions})
future_data.set_index('Date', inplace=True)


full_data = pd.concat([data['Close'], future_data['Close']])
full_data = full_data.asfreq('D')
full_data = full_data.interpolate(method='time')


plt.figure(figsize=(12, 6))
sns.lineplot(data=full_data, label='Historical & Future Predictions')
plt.title('Stock Price Prediction with PCA')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()


# In[ ]:




