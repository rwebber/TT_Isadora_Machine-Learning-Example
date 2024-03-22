# TT Isadora ML example

This example file offers users a quick and easy way to collect numeric data bundles and create a Machine Learning trained model from the sample data, that can be used to classify additional data bundles in future Isadora scenes.

The example Isadora file starts with a training scene where X and Y coordinates are input as data samples along with a classifier. Once ample samples are recorded, this training data is processed into a trained model.

The next scene of the Isadora file uses the trained model to predict the classification of real-time incoming X and Y coordinates.

This rudimentary example illustrates the flexible nature of the JSON-based input bundles, which can be extended incredibly easily.

### REQUIREMENTS

* Isadora 4.x
* Pythoner actor (released with Isadora 4)
* Creation of a Python Virtual Environment via the requirements.txt file
