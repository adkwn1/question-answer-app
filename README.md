## Question-Answer Streamlit App Using T5 Models
Author: Andrew Kwon

## Introduction

This project was created as part of TripleTen's externship program in collaboration with DataSpeak. Project is still a work in progress, but currently functional. The application showcases three concepts for a question-answer model using pre-trained and fine-tuned T5 models. The dataset used for context and model training was dervied from approximately 100,000 Python questions from Stack Overflow. The tasks are preformed as follows:

Model 1: Retrieval Augmented Generation (RAG) Question-Answering
- Takes a user query and retrieves K-similar semantic matches
- The retrieved responses are then passed to:
  1) A fine-tuned T5 conditional generation model (available on HuggingFace: 'c-kilo-1/results') to generate an answer.
  2) A pre-trained T5 model (t5-base) to summarize the retrieved responses based on K-similar returns.

Model 2: Text-Generation
- Takes a user query and provides an answer generated using a fine-tuned T5 conditional generation model ('c-kilo-1/results').

The task for all three cases are designed to provide closed-book answers in order to provide appropriate responses to domain specific questions. For proof of concept purposes, the domain in this case are questions related to Python based on community provided topics and responses on Stack Overflow. As such, validity of answers may vary.

## Dataset
The dataset can be downloaded from Kaggle (https://www.kaggle.com/datasets/stackoverflow/pythonquestions). The provided notebook (nb_parser.ipynb) was used to clean and merge the dataset, and export the final .csv file(s) for usage.

## Usage - Local Browser

To run the project in a local web browser, clone the repository ensuring the files are in the correct directory structure:
- app_combined.py (top level)
- .streamlit/config.toml
- datasets/final_sample.csv (created via notebook from original dataset)

The config.toml specifies the server address and port number to run the web app locally via streamlit. In command prompt or terminal, navigate to the cloned directory and run the following command:

<code>streamlit run app_combined.py</code>

Open a web browser and navigate to http://localhost:10000 to interact with the project application.

## Requirements
- pandas
- numpy
- pickle
- streamlit
- transformers
- sentence_transformers

## Screenshots

![qa_model2](https://github.com/adkwn1/question-answer-app/assets/119823114/d463b05d-adad-427e-bcb1-3b2af3645dfc)
![qa_model](https://github.com/adkwn1/question-answer-app/assets/119823114/3d17900d-c561-4b3d-a9b9-4b04e772ce02)
