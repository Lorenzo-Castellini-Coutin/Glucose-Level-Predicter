#Author: Lorenzo Castellini Coutin

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

GlucoseTable = pd.read_csv()   

columns_delete = [
    'Index', 'Transmitter ID', 'Transmitter Time (Long Integer)', 'Glucose Rate of Change (mg/dL/min)','Carb Value (grams)', 'Event Type',
    'Patient Info', 'Duration (hh:mm:ss)', 'Device Info', 'Source Device ID', 'Event Subtype', 'Insulin Value (u)'
]

GlucoseTable = GlucoseTable.drop(columns = columns_delete)

GlucoseTable.columns = ['Dates', 'GlucoseLvls']    

GlucoseTable['Dates'] = GlucoseTable['Dates'].astype('string')
GlucoseTable['Dates'] = GlucoseTable['Dates'].str.replace('T', '')
GlucoseTable['Dates'] = GlucoseTable['Dates'].str[:10] + ' ' + GlucoseTable['Dates'].str[10:]

GlucoseTable = GlucoseTable[GlucoseTable['GlucoseLvls'] != 0.0]
GlucoseTable = GlucoseTable[GlucoseTable['GlucoseLvls'] != 'Low']
GlucoseTable = GlucoseTable[GlucoseTable['GlucoseLvls'] != 'High']

GlucoseTable = GlucoseTable.dropna()

GlucoseTable['GlucoseLvls'] = GlucoseTable['GlucoseLvls'].astype('float')
GlucoseTable['Dates'] = GlucoseTable['Dates'].astype('datetime64[ns]')

def lag_features(df, lags): 
    df_lagged = df.copy()
    for lag in lags:
        df_lagged[f'lag_{lag}'] = df_lagged['GlucoseLvls'].shift(lag)
    return df_lagged

lags_num = [1, 2, 3, 4, 5, 6, 7]  
GlucoseTable = lag_features(GlucoseTable, lags_num)
GlucoseTable = GlucoseTable.dropna()

# Split data into features and target
X = GlucoseTable[['lag_' + str(lag) for lag in lags_num]]
Y = GlucoseTable['GlucoseLvls']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.1, shuffle = False)


model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    eval_metric='mae',
    booster = 'dart',
    learning_rate = 0.1,
    max_depth = 4,
    min_child_weight = 4, 
    subsample = 0.7,
    n_estimators = 70
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')


plt.plot(GlucoseTable['Dates'].iloc[-len(y_pred):], y_pred, label='Predicted Values', color='red')
plt.plot(GlucoseTable['Dates'], GlucoseTable['GlucoseLvls'], label = '', color = 'blue')
plt.xlabel('Dates (Year-Month-Day)')
plt.ylabel('Glucose Values (mg/dL)')
plt.title('Glucose Values Forecast')
plt.legend()
plt.grid(True)
window = plt.get_current_fig_manager()
window.full_screen_toggle()
plt.show()


#Author: Lorenzo Castellini Coutin
