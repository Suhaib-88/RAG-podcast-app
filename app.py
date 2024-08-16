import streamlit as st
from groq import Groq
import os
import pandas as pd
import numpy as np
from pydub import AudioSegment

import pydub
from langchain.text_splitter import TokenTextSplitter
from src.transcription import documentation
from src.chunking import dataframe_chunking
from src.vector_store import create_vector_store, retrieve_vector_store

pydub.AudioSegment.ffmpeg = 'ffmpeg.exe'
pydub.AudioSegment.ffmpeg = 'ffprobe.exe'
pydub.AudioSegment.ffmpeg = 'ffplay.exe'


os.environ['PINECONE_API_KEY']="60815dfe-f4a7-4a4d-84fa-acc0415803c6"

def transcript_chat_completion(client, transcript, user_query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": '''Use this transcript or transcripts to answer any user questions, citing specific quotes:

                {transcript}
                '''.format(transcript=transcript)
            },
            {
                "role": "user",
                "content": user_query,
            }
        ],
        model="llama3-8b-8192",
    )

    return chat_completion.choices[0].message.content

def main():
    st.sidebar.title("API key config")
    api_key = st.sidebar.text_input("Enter your Api key", type='password')
    st.title("AI Voice assistant")
    st.write("This is a simple voice assistant that can perform various tasks")

    if api_key:
        try:
            with st.sidebar:
                st.success("API key successfully set ✅")
            client = Groq(api_key=api_key)
            whisper_model = 'whisper-large-v3'
            text_splitter = TokenTextSplitter(
                chunk_size=500,
                chunk_overlap=20
            )

            mp3_file_folder = "mp3-files"
            mp3_chunk_folder = "mp3-chunks"
            chunk_length_ms = 1000000
            overlap_ms = 10000

            episodes_df = dataframe_chunking(mp3_file_folder, mp3_chunk_folder, chunk_length_ms, overlap_ms)
            documents = documentation(episodes_df, text_splitter, client, whisper_model)

            pinecone_index_name = "twist-transcripts"
            docsearch = create_vector_store(documents, pinecone_index_name)

            user_question = st.chat_input("Chat with the audio files")
            if user_question:
                with st.chat_message('user'):
                    st.write("Query: ", user_question)

                with st.chat_message('assistant'):
                    if user_question is not None:
                        docsearch=retrieve_vector_store(index_name="twist-transcripts")
                        relevant_docs = docsearch.similarity_search(user_question)
                        relevant_transcripts = '\n\n------------------------------------------------------\n\n'.join(
                            [doc.page_content for doc in relevant_docs[:3]])
                        ai_response = transcript_chat_completion(client, relevant_transcripts, user_question)
                        st.write(ai_response)
                    else:
                        st.info("We are Ready, Please start your conversation")

        except Exception as e:
            raise e

    else:
        with st.sidebar:
            st.error("Set your API token first 🚩")

if __name__ == "__main__":
    main()