## Question-Answer Streamlit App Using T5 Models
Author: Andrew Kwon

## Introduction

This project was created as part of TripleTen's externship program in collaboration with DataSpeak. Project is still a work in progress, but currently functional. The application showcases three concepts for a question-answer model using pre-trained and fine-tuned T5 models. The dataset used for context and model training was dervied from approximately 100,000 Python questions from Stack Overflow. The tasks are preformed as follows:

Model 1: Retrieval Augmented Generation (RAG) Question-Answering
- Takes a user query and retrieves K-similar semantic matches
- The retrieved responses are then passed to:
  1) A pre-trained T5 model (t5-base) to summarize the retrieved responses based on K-similar returns.
  2) A fine-tuned T5 conditional generation model (available on HuggingFace: 'c-kilo-1/t5-sm-py-stackoverflow') to generate an answer.

Model 2: Text-Generation
- Takes a user query and provides an answer generated using a fine-tuned T5 conditional generation model ('c-kilo-1/t5-flan-py-stackoverflow').

The task for all three cases are designed to provide closed-book answers in order to provide appropriate responses to domain specific questions. For proof of concept purposes, the domain in this case are questions related to Python based on community provided topics and responses on Stack Overflow. As such, validity of answers may vary.

## Dataset
Added a small sample of the final dataset for demonstrative purposes (15,000 entries, ~20 MB). 

The main dataset can be downloaded from Kaggle (https://www.kaggle.com/datasets/stackoverflow/pythonquestions). The provided notebook (nb_parser.ipynb) was used to clean and merge the dataset, and export the final .csv file(s) for usage.

## Usage - Local Browser

To run the project in a local web browser, clone the repository ensuring the files are in the correct directory structure:
- app_demo.py (top level)
- .streamlit/config.toml
- datasets/final_sample.csv (created via notebook from original dataset)
  -  Can use 'datasets/final_sample_small.csv' for limited demo

The config.toml specifies the server address and port number to run the web app locally via streamlit. In command prompt or terminal, navigate to the cloned directory and run the following command:

<code>streamlit run app_demo.py</code>

Open a web browser and navigate to http://localhost:10000 to interact with the project application.

## Requirements
- pandas
- numpy
- pickle
- streamlit
- transformers
- sentence_transformers

## Screenshots

![screenshot](https://github.com/adkwn1/question-answer-app/assets/119823114/9cc8bd4c-c483-45b8-9257-362e9603aa27)

