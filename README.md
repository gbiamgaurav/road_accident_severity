# End to End Project on Accident Severity 


This is a multiclass classification project to classify severity of road accidents into three categories. this project is based on real-world data. Learn more about detailed description and problem with tasks performed.

**Description**: This data set is collected from Addis Ababa Sub-city, Ethiopia police departments for master's research work. The data set has been prepared from manual records of road traffic accidents of the year 2017-20. All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident. Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms.

**Dataset source**: [click to check data source](https://www.narcis.nl/dataset/RecordID/oai%3Aeasy.dans.knaw.nl%3Aeasy-dataset%3A191591)

**Problem Statement**: The target feature is Accident_severity which is a multi-class variable. The task is to classify this variable based on the other 31 features step-by-step by going through each day's task. Your metric for evaluation will be f1-score.


## Steps 

1. Create environment

`conda create -p rtacls python==3.8 -y`

2. Run setup.py and install requirements.txt

`pip install -r requirements.txt`

3. To run the app

`streamlit run app.py`

**app**: [find the app here](http://accidentseverity-env.eba-9zrk9m6h.us-east-1.elasticbeanstalk.com/)

