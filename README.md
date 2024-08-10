## Glucose Level Predicter
### Introduction & Scope
This Python script uses the Dexcom Overview Report csv file as its dataset, in order to train an ML model in order to predict future glucose levels. The current and future glucose levels are represented in a graph. The graph can help the patient know where they will roughly stand with regards to their glucose levels in the near future which can help adjust their insulin dosage, meals, among other aspects that affect blood sugar levels. In the end, this can all help the user lower their h1ac.
### How it works?
The script's ML model is currently under remodeling.... As of now, the script is able to make the forecast for the future glucose levels, using the Neural Prophet model. Nonetheless, the current model/forecast is not fully accurate since it is not capable of detecting peaks.
### Usage
In order to run this script locally in your machine, you can clone the repository. Make sure you have downloaded all of the used libraries. Furthermore, make sure you have the required dataset, in this case the Dexcom Overview Report csv available. Laslty, the script is only compatible with Dexcom CGM Overview Report csv, and might not work with other CGMs from other companies. 
### Advisory on Usage
This script gives its users a general idea about their glucose levels, but has its limitations. This predicter does not replace a professional's adivce, therefore, for more detailed and accurate information please visit your doctor. 
### Additional Notes
The Dexcom Overview Report csv files contains confidential information with regards to a patient's glucose levels, therefore, beware before using the script for your own glucose levels, since the model uses said levels to form a prediction. Furthermore, this script is not associated with the Dexcom company. Currently, the model is being reworked and other options for ML models are being explored in order to improve accuracy. For more up-to-date changes, checkout the 'updates' branch, where the bug fixes and implementation of new ML models will take place. In the 'main' branch, certain versions will be stred, until a new version is done and fully tested. 
