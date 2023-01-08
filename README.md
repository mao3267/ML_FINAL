# ML Final Project - Tabular Playground Series Aug 2022

This repository is my implementation of [TPS - Aug 2022](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview). 

### My Results (Late Submission)
![](https://i.imgur.com/IrSKuig.png)
## Requirements

### On Google Colab (Recommended)

You need to get your Kaggle API Token (kaggle.json) first. The tutorial is [here](https://www.kaggle.com/general/74235).
After you start to run the whole notebook, the third block will ask you to upload your token.

### Local 
My local environment: MacOS M1 Ventura
with a virtualenv by running:
```zsh
brew install virtualenv
```
Create a new virtual environment
```zsh
virtualenv -p <your-python-path> ml_final
```
Enter the virtual environment
```zsh
. ./ml_final/bin/activate
```
To install requirements:
```zsh
pip install -r requirements.txt
```

## Training

### On Google Colab
To train the model(s), click on Runtime -> Run all:


### Local 

#### Before Training
- Download the data from [here](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data) and unzip them to `./train` by
```zsh
unzip ~/Downloads/tabular-playground-series-aug-2022.zip -d train
```
#### Start Training
Remember to comment the kaggle upload code and run all
### Features

You can test your own feature sets by changing the `feature_used` list and set up the corresponding features at the `Feature Engineering` block.

## Evaluation
### Before evaluation
- Store the models as `model_i.pkl` in `./model` under your working directory. 
- Download the data from [here](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data) and unzip them to `./train` by
```zsh
unzip ~/Downloads/tabular-playground-series-aug-2022.zip -d train
```
### On Google Colab

The notebook will train and then evaluate at the `submission` section. If you want to evaluate on your own models, use the `inference.ipynb`.

### Local

Activate the virtual environment and run all in `109550003_Final_inference.ipynb` to evaluate the models and generate a submission.


- All the preprocess on test.csv is done in the Jupyter Notebook
- If you want to use your own preprocessed data, comment the function `preprocess`. 
- You can submit the submission.csv to the Kaggle as late submission to check your score.
- Take a look at the Kaggle LeaderBoard [here](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/leaderboard)

## Pre-trained Models

You can download pretrained models here:

- [5 Models for ensemble](https://drive.google.com/drive/folders/11daP6XIH65Hw24mCHLO25GiEf7y93gWQ?usp=sharing). 

## Results

My model achieves the following performance on the test data:

| Model name         | Private Score   | Public Score   |
| ------------------ |---------------- | -------------- |
| My Model           |     0.59141     |      0.58964   |


## References
- Feature engineering based on [TPSAUG22 EDA which makes sense ⭐️⭐️⭐️⭐️⭐️](https://www.kaggle.com/code/ambrosm/tpsaug22-eda-which-makes-sense#The-float-columns)
- Impute measurement_17 based on [Perfect Positive Correlation with measurement_17](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/discussion/343939)
- Scale the data according to [Adversarial validation - detecting data drift](https://www.kaggle.com/code/nnjjpp/adversarial-validation-detecting-data-drift)
- HuberRegression and measurement_17 from [tps-aug22-9th-solution](https://www.kaggle.com/code/takanashihumbert/tps-aug22-9th-solution/notebook)
- Weight of Evidence from [Combine LogisticRegression](https://www.kaggle.com/code/argyrisanastopoulos/private-score-0-59144-combine-logisticregression)
