import pandas as pd
import numpy as np
import pickle
import streamlit as st

from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import util, SentenceTransformer

df = pd.read_csv('datasets/final_sample_small.csv')

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
gen_ft_model = T5ForConditionalGeneration.from_pretrained('t5-flan-py-stackoverflow')   # locally saved
rag_ft_model = T5ForConditionalGeneration.from_pretrained('c-kilo-1/t5-sm-py-stackoverflow')
rag_pt_model = T5ForConditionalGeneration.from_pretrained('t5-base')

try:
    embed_file = open('final_small_title_answer.obj', 'rb')
    title_embeddings = pickle.load(embed_file)
    embed_file.close()
except:
    title_embeddings = embed_model.encode(df['title_answer'], convert_to_tensor=True).cpu()
    embed_file = open('final_small_title_answer.obj', 'wb') 
    pickle.dump(title_embeddings, embed_file)
    embed_file.close()

def getSimiliar(input, k):
    input_embedding = embed_model.encode(input, convert_to_tensor=True).cpu()
    cos_score = (util.cos_sim(input_embedding, title_embeddings)).numpy()[0]

    indices = sorted(range(len(cos_score)), key=lambda x: cos_score[x])[-k:]
    indices.reverse()
 
    res = df.iloc[indices]
    res['CosSim'] = cos_score[indices]

    text = ''
    for i in res['Response']:
        text += ' ' + i
    text = text.strip()
    
    return res, text

# Function for generating text output given a corpus of specific answers retrieved from the dataset.
def generateOutput(prompt, model, task):
    input = task + prompt   # 'generate_answer: text strings'
    encoded_input = tokenizer(input, truncation=True, return_tensors='pt')
    output = model.generate(input_ids = encoded_input.input_ids, max_length=300, repetition_penalty=1.5)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

st.title('DataSpeak Externship Project')
query = st.text_input(label='Please ask a question or describe your issue:', value="", max_chars=200, key=1)

k = 3
threshold = 0.5
response, text = getSimiliar(query, k)

if response.iloc[0]['CosSim'] < threshold:
    st.write(f'No similar queries found (threshold {threshold}).')    
else:
    st.header('RAG (Pre-trained + Summarize):')
    st.write(generateOutput(text, rag_pt_model, 'summarize: '))
    
    st.header('RAG (Fine-tuned):')
    st.write(generateOutput(response.iloc[0]['title_answer'], rag_ft_model, 'generate_answer: '))

#####
st.header('', divider='blue')
#####

st.header('Generate Response (Fine-tuned):')
st.write(generateOutput(query, gen_ft_model, 'answer the question: '))

#####

st.header('', divider='blue')
st.write(':grey[Source:]')
for idx, row in response.iterrows():
    st.write(f":grey[{row['CosSim']:.2f} | {row['Title']}]")
    st.write(f":grey[{row['Response']}]")