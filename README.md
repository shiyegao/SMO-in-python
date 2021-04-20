# SMO in Python and Cross Validation
+ Simple SMO in python, able to generate data, train, test, and compare with svm.SVC
+ Cross validation codes for comparison of different models, using metrics like accuracy and F1 score 

# Installation
+ Clone this repo:
```bash
git clone https://github.com/shiyegao/SMO-in-python.git
cd SMO-in-python
```
+ Install some dependencies:
```bash
pip install -r requirements.txt
```

# Getting started
## run SMO
```bash
python smo.py
```
We have prepared some pretrained models(parameters) in (/data/parameter_*)[https://github.com/shiyegao/SMO-in-python/tree/master/data], you can change some codes in SMO.py to just test the pretrained model.
## run Cross Validation
```bash
python cross_val.py
```
We use SVM and LogisticRegression for comparison. You can change the **models** and **model_name** in cross_val.py to compare more different models.

The metrics are accuracy and F1 score. If you have more suggestions, welcome pull request!
