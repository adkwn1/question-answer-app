import pandas as pd
import numpy as np
import pickle
import streamlit as st

from transformers  import  AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import util, SentenceTransformer

df = pd.read_csv('datasets/final_sample.csv')

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
rag_tokenizer = AutoTokenizer.from_pretrained('t5-base')
rag_model = T5ForConditionalGeneration.from_pretrained('c-kilo-1/results')
rag_model2 = AutoModelForSeq2SeqLM.from_pretrained('t5-base')

try:
    embed_file = open('embed_file.obj', 'rb')
    title_embeddings = pickle.load(embed_file)
    embed_file.close()
except:
    title_embeddings = embed_model.encode(df['title_body'], convert_to_tensor=True).cpu()
    embed_file = open('embed_file.obj', 'wb') 
    pickle.dump(title_embeddings, embed_file)
    embed_file.close()

def getSimiliar(input, k):
    input_embedding = embed_model.encode(input, convert_to_tensor=True).cpu()
    cos_score = (util.cos_sim(input_embedding, title_embeddings)).numpy()[0]

    indices = sorted(range(len(cos_score)), key=lambda x: cos_score[x])[-k:]
    indices.reverse()
 
    res = df.iloc[indices][['ParentId', 'title_body', 'Response', 'Score']]
    res['CosSim'] = cos_score[indices]

    text = ''
    for i in res['Response'].to_list():
        text += ' '+ i
    text = text.strip()
    
    return res, text

# Function for generating text output given a corpus of specific answers retrieved from the dataset.
# Currently uses pre-trained t5-base model for tokenizer and text generator
def generateOutput(prompt):
    input = f'generate_answer: {prompt}'
    encoded_input = rag_tokenizer(input, 
                              truncation=True,
                              return_tensors='pt')
    output = rag_model.generate(input_ids = encoded_input.input_ids,
                            attention_mask = encoded_input.attention_mask,
                            max_length=200,
                            repetition_penalty=1.5)
    return rag_tokenizer.decode(output[0], skip_special_tokens=True)

def summarizeOutput(context):
    input = f'summarize: {context}'
    encoded_input = rag_tokenizer(input, 
                              truncation=True,
                              return_tensors='pt')
    output = rag_model2.generate(input_ids = encoded_input.input_ids,
                            attention_mask = encoded_input.attention_mask,
                            max_length=200,
                            repetition_penalty=1.5)
    return rag_tokenizer.decode(output[0], skip_special_tokens=True)

st.title('DataSpeak Externship Project')
st.header('Model 1: RAG QA')
k = st.slider('K-similar documents', 1, 5)

rag_query = st.text_input(label='Please ask a question or describe your issue:', value="", max_chars=200, key=1)
response, text = getSimiliar(rag_query, k)

st.subheader('T5 Fine-Tuned:')
st.write(generateOutput(response['title_body'].iloc[0]))
st.subheader('T5 Pre-Trained:')
st.write(summarizeOutput(text))

#####
st.header('', divider='blue')
#####

st.header('Model 2: Fine-Tuned T5 Conditional Generation')

pt_tokenizer = T5Tokenizer.from_pretrained('t5-small')
pt_model = T5ForConditionalGeneration.from_pretrained('c-kilo-1/results')
task_prefix = 'generate_answer: '

pt_query = st.text_input(label='Please ask a question or describe your issue:', value="", max_chars=200, key=2)

input_ids = pt_tokenizer(task_prefix + pt_query, return_tensors="pt").input_ids
outputs = pt_model.generate(input_ids,
                         max_length=200,
                         repetition_penalty=1.5)

st.write(pt_tokenizer.decode(outputs[0], skip_special_tokens=True))