#Author: Lorenzo Castellini Coutin

import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet

GlucoseTable = pd.read_csv()   #Insert the csv file here

GlucoseTable.pop('Index')
GlucoseTable.pop('Transmitter ID')
GlucoseTable.pop('Transmitter Time (Long Integer)')
GlucoseTable.pop('Glucose Rate of Change (mg/dL/min)')
GlucoseTable.pop('Carb Value (grams)')
GlucoseTable.pop('Event Type')
GlucoseTable.pop('Patient Info')
GlucoseTable.pop('Duration (hh:mm:ss)')
GlucoseTable.pop('Device Info')
GlucoseTable.pop('Source Device ID')
GlucoseTable.pop('Event Subtype')
GlucoseTable.pop('Insulin Value (u)')

GlucoseTable.columns = ['ds', 'y']    #ds is dates and y is glucose values

GlucoseTable['ds'] = GlucoseTable['ds'].astype('string')
GlucoseTable['ds'] = GlucoseTable['ds'].str.replace('T', '')
GlucoseTable['ds'] = GlucoseTable['ds'].str[:10] + ' ' + GlucoseTable['ds'].str[10:]

GlucoseTable['y'] = GlucoseTable['y'].astype('string')
GlucoseTable['y'] = GlucoseTable['y'].replace('Low', float('nan'))
GlucoseTable['y'] = GlucoseTable['y'].replace('0', float('nan'))
GlucoseTable['y'] = GlucoseTable['y'].replace('High', float('nan'))

GlucoseTable = GlucoseTable.dropna()

GlucoseTable['y'] = GlucoseTable['y'].astype('float')
GlucoseTable['ds'] = GlucoseTable['ds'].astype('datetime64[ns]')

model = NeuralProphet(
    learning_rate = 0.1,
    growth = 'linear',
    epochs = 130,
    changepoints_range = 20
)

GlucoseValTest = model.fit(GlucoseTable)

Future = model.make_future_dataframe(GlucoseTable, periods = 2000)

FutureVals = model.predict(Future)

Model_OriginalVals_Pred = model.predict(GlucoseTable)

plt.title('Glucose Values (mg/dL) vs. Days')
plt.xlabel('Days')
plt.ylabel('Glucose Values (mg/dL)')
plt.plot(GlucoseTable['ds'], GlucoseTable['y'], label = 'Original Glucose Values', c ='g')
plt.plot(FutureVals['ds'], FutureVals['yhat1'], label = 'Future Glucose Values', c = 'r')
plt.plot(Model_OriginalVals_Pred['ds'], Model_OriginalVals_Pred['yhat1'], label = 'ML Model Predicted Glucose Values', c = 'b')
plt.legend(loc = 'upper center')

window = plt.get_current_fig_manager()
window.full_screen_toggle()

plt.show()




#Author: Lorenzo Castellini Coutin
