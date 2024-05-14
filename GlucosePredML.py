import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet

from datetime import datetime

GlucoseTable = pd.read_csv(r'C:\Users\loren\Downloads\glucosevalues.csv')

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

GlucoseTable = GlucoseTable.dropna()

GlucoseTable['Timestamp (YYYY-MM-DDThh:mm:ss)'] = GlucoseTable['Timestamp (YYYY-MM-DDThh:mm:ss)'].astype("string")
GlucoseTable['Timestamp (YYYY-MM-DDThh:mm:ss)'] = GlucoseTable['Timestamp (YYYY-MM-DDThh:mm:ss)'].str.replace('T', '')

GlucoseTable['Timestamp (YYYY-MM-DDThh:mm:ss)'] = GlucoseTable['Timestamp (YYYY-MM-DDThh:mm:ss)'].str[:10] + ' ' + GlucoseTable['Timestamp (YYYY-MM-DDThh:mm:ss)'].str[10:]
GlucoseTable.columns =['ds', 'y']

model = NeuralProphet()
GlucoseValTest = model.fit(GlucoseTable)

Future = model.make_future_dataframe(GlucoseTable, periods = 200)

Forecast = model.predict(Future)

