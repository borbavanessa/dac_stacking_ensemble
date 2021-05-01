## Important

To comply with the Data Usage Agreement, we removed the pre-trained models (weak classifiers and DAC Stacking models) together with the tokenizers and embeddings from the domain. 
This is because the content of the terms can be partially retrieved through this file. Even so, to run this project, make sure to generate the following directory structure:

stacking_ensemble
|__ dataset
|__ tokenizers
|__ pre_train_embeddings
|__ network_model
|__ utils
|__ weak_classifiers
|__ experiments/data_mining

Content:
dataset - files with dataset in format Dataframe Pandas
tokenizers - text content resulting to convertion for tokenization process
pre_train_embeddings - file of the embeddings pre-train
network_model - source code generate weak classifiers and stacking ensemble model
utils - source code with auxiliary functions to generate models and pre-process data
weak_classifiers - must contain the models generated for the weak classifiers
experiments/data_mining - jupyters notebooks set to run considering from that directory path
