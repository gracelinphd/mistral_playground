import streamlit as st
import pandas as pd
import numpy as np
import lancedb
import requests
import json
import os

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


# Initialize the Mistral client
mistral_api_key = os.getenv('MISTRAL_API_KEY')
client = MistralClient(api_key=mistral_api_key)

# Read tarot/oracle files
def get_data(file_path):
  df = pd.read_csv(file_path)

  # Convert to records so each row is a dictionary, and it's a list of dictionaries
  df_dict = df.to_dict(orient='records')

  # Combine keys and values into a single string
  texts = []
  for d in df_dict:
    text = ''
    for key, value in d.items():
      text += f"{key}: {value}; "
    texts.append(text.strip())

  return df_dict, texts

# The gentle tarot
file_path = 'data/prisma_visions_tarot.csv'
gentle_tarot_df_dict, gentle_tarot_texts = get_data(file_path)

# Bloom oracle
file_path = 'data/your_wise_animal_body_oracle.csv'
bloom_oracle_df_dict, bloom_oracle_texts = get_data(file_path)

# Combine texts
texts = gentle_tarot_texts + bloom_oracle_texts
df = pd.DataFrame(texts, columns=['text'])

# Text embedding
def get_text_embedding(input):
    embeddings_batch_response = client.embeddings(
          model="mistral-embed",
          input=input
      )
    return embeddings_batch_response #.data[0].embedding

# Get embeddings in batches
# Might take some time
def get_embeddings_by_chunks(data, chunk_size=50):
    chunks = [data[x:x + chunk_size] for x in range(0, len(data), chunk_size)]
    embeddings_response = [get_text_embedding(c) for c in chunks]
    return [d.embedding for e in embeddings_response for d in e.data]

# Get embeddings for all rows
df["vector"] = get_embeddings_by_chunks(texts)

# Create LanceDB database and table
db = lancedb.connect("./lancedb")
table = db.create_table("tarot_cards", data=df, mode="overwrite")

# Funtion to chat with Mistral
def run_mistral(user_message, model="mistral-large-latest"):
    messages = [
        ChatMessage(role="user", content=user_message)
    ]
    chat_response = client.chat(
        model=model,
        messages=messages
    )
    return (chat_response.choices[0].message.content)


# Streamlit app
st.title("Tarot Reading App with Mistral AI RAG")

# User input
query = st.text_area("Enter your query with tarot/oracle cards:",
					 height=150,
					 help="You can enter multiple lines of text. Include your question and any specific cards you want to ask about.")

# Prompts

# identify the cards mentioned
identify_cards_prompt = """
Identify all the cards in the message below. They may be tarot or oracle cards. They may be non-standard tarot cards.
Show the cards only, in comma-separated format, no notes or any other information.

Message: """

# query with retrieved info
rag_prompt = """
Context information is below.
---------------------
{}
---------------------
Answer the query by interpreting cards and incorportaing:
- context information above
- notable synchronicities on the objects of card images
- notable synchronicities on colors of the card images
- notable synchronicities on the numbers of the cards
- any other notable features of these cards

The final interpretation should incorporation all individual cards' interpretations.

Query: {}
Answer:
"""

if st.button("Get Reading"):
    if query:
        with st.spinner("Fetching cards information..."):
            # Identify the cards mentioned
            response = run_mistral(identify_cards_prompt+query)
            cards = [c.strip() for c in response.split(',')]
            st.write("Identified cards:", cards)

            # Get embeddings of requested cards
            cards_embeddings = np.array([get_text_embedding(card).data[0].embedding for card in cards])

            # Retrieve similar chunks from LanceDB
            retrieved_chunks = []
            for embedding in cards_embeddings:
                results = table.search(embedding).limit(2).to_pandas()
                # only take the first result of each card retrieved
                retrieved_chunks.append(results['text'][0])

            response = run_mistral(rag_prompt.format('\n'.join(retrieved_chunks), query),
            						model="mistral-large-latest")
            st.success(response)

            # Display retrieved chunk
            st.subheader("Cards Retrieved")
            st.json(retrieved_chunks)
    else:
    	st.warning("Please enter a query before getting a reading.")

