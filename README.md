## Glucose Level Predicter
### Introduction & Scope
This Python script uses the Dexcom Overview Report csv file as its dataset, in order to train an ML model in order to predict future glucose levels. The current and future glucose levels are represented in a graph. The graph can help the patient know where they will roughly stand with regards to their glucose levels in the near future which can help adjust their insulin dosage, meals, among other aspects that affect blood sugar levels. In the end, this can all help the user lower their h1ac.
### How it works?
The script's model is currently under remodeling....
### Usage
In order to run this script locally in your machine, you can clone the repository. Make sure you have downloaded all of the used libraries. Furthermore, make sure you have the required dataset, in this case the Dexcom Overview Report csv available. Laslty, the script is only compatible with Dexcom CGM Overview Report csv, and might not work with other CGMs from other companies. 
### Advisory on Usage
This script gives its users a general idea about their glucose levels, but has its limitations. This predicter does not replace a professional's adivce, therefore, for more detailed and accurate information please visit your doctor. 
### Additional Notes
Currently working on ways to improve the accuracy of the ML model. I am comparing accuracy by analyzing the ML model's predictions on the past glucose levels, in order to compare them with the original/accurate levels. In this process, I am seeking to make the current ML model more accurate by adjusting the hyperparameters and/or looking for other prebuilt ML models which fit my use case. Furthermore, looking at other features to implement...
