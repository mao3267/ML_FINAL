# ML Final Project - Tabular Playground Series Aug 2022

This repository is my implementation of [TPS - Aug 2022](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview). 

### My Results (Late Submission)
>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

### Run on Google Colab (Recommended)

You need to get your Kaggle API Token (kaggle.json) first. The tutorial is [here](https://www.kaggle.com/general/74235).
After you start to run the whole notebook, the third block will ask you to upload your token.

### Local 
To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

### On Google Colab
To train the model(s), click on Runtime -> Run all:


### Features

You can test your own feature sets by changing the `feature_used` list and set up the corresponding features at the `Feature Engineering` block.

## Evaluation

To evaluate my model on test data and generate a submission, run:

```eval
python eval.py 
```

- You can submit the submission.csv to the Kaggle for late submission and check your score.
- Take a look on the Kaggle LeaderBoard [here](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/leaderboard)

## Pre-trained Models

You can download pretrained models here:

- [My model](https://drive.google.com/mymodel.pth). 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

My model achieves the following performance on the test data:

| Model name         | Private Score   | Public Score   |
| ------------------ |---------------- | -------------- |
| My Model           |     0.59108     |      0.5897    |


## References
- Feature engineering based on [TPSAUG22 EDA which makes sense ⭐️⭐️⭐️⭐️⭐️](https://www.kaggle.com/code/ambrosm/tpsaug22-eda-which-makes-sense#The-float-columns)
- Impute measurement_17 based on [Perfect Positive Correlation with measurement_17](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/discussion/343939)
- Scale the data according to [Adversarial validation - detecting data drift](https://www.kaggle.com/code/nnjjpp/adversarial-validation-detecting-data-drift)
- WoE and HuberRegression from [tps-aug22-9th-solution](https://www.kaggle.com/code/takanashihumbert/tps-aug22-9th-solution/notebook)
