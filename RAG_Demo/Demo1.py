# # from langchain_community.document_loaders import UnstructuredURLLoader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_chroma import Chroma
# # from langchain_openai import OpenAIEmbeddings, OpenAI
# # import streamlit as st
# # import time
# # from dotenv import load_dotenv

# # from langchain.chains import create_retrieval_chain
# # from langchain.chains.combine_documents import create_stuff_documents_chain
# # from langchain_core.prompts import ChatPromptTemplate
# # import os

# # load_dotenv()

# # st.title("RAG Application")
# # api_key = os.getenv("OPENAI_API_KEY")
# # # st.write(f"API Key: {api_key}")  # Debugging line

# # if api_key is None:
# #     st.error("OPENAI_API_KEY environment variable not set.")
# #     st.stop()
# # os.environ["OPENAI_API_KEY"] = api_key

# # urls = ["https://en.wikipedia.org/wiki/English_Wikipedia"]
# # loader = UnstructuredURLLoader(urls)
# # data = loader.load()
# # # for doc in data:
# # #     print(doc.page_content
# # #split data into chunks
# # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
# # split_data = text_splitter.split_documents(data)

# # # Embedding convert data intovectors(numerical representation)
# # # store vectors into a vector database using chroma
# # vectorstore = Chroma.from_documents(split_data, OpenAIEmbeddings())

# # # retriver will retrive the information from the vector database. 
# # # We use cosine similarity to search for similar vectors and fetch the 5 most similar datas

# # retriever = vectorstore.as_retriever(search_type="cosine_similarity",search_kwargs={"n_results":5})
# # retrieved_docs = retriever.invoke("What kind of services they provide?")

# # # Use LLM to generate response
# # # Temperature will set the randomness of the response. 
# # # close to 1 increase the diverse and random response and vice versa
# # llm = OpenAI(temperature=0.5, max_tokens=500)

# # # chatbox in page
# # query = st.chat_input("tell something")
# # prompt = query
# # # system prompt will guide the LLm to generate the response using the context in a format
# # system_prompt = (
# #     "You are an assitant for question answering tasks",
# #     "Use the selected contexts of the retrieved context to answer the question",
# #     "If you dont know the answer to the question, you can say 'I dont know'",
# #     "Use 3 sentences maximum to keep answer concise",
# #     "\n\n"
# #     "{context}"
# # )

# # prompt = ChatPromptTemplate.from_messages(
# #     [
# #         ("system", system_prompt),
# #         ("human", "{input}")
# #     ]
# # )


# # if query:
# #     question_answer_chain = create_stuff_documents_chain(retrieved_docs, llm, prompt)
# #     rag_chain = create_retrieval_chain(retriever, question_answer_chain)    
# #     result = rag_chain.invoke({"input": query})
# #     st.write(result["answers"])

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Load environment variables
load_dotenv()

# Debugging: Check if the API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
st.write(f"API Key: {api_key}")  # Debugging line
if api_key is None:
    st.error("OPENAI_API_KEY environment variable not set.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# Streamlit app title
st.title("RAG Application")

# Load data from URLs
urls = ["https://en.wikipedia.org/wiki/English_Wikipedia"]
loader = UnstructuredURLLoader(urls)
data = loader.load()

# Split data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
split_data = text_splitter.split_documents(data)

# Embedding and storing in Chroma
vectorstore = Chroma.from_documents(split_data, OpenAIEmbeddings())

# Retriever setup
retriever = vectorstore.as_retriever(search_type="cosine_similarity", search_kwargs={"n_results": 5})

# LLM setup
llm = OpenAI(temperature=0.5, max_tokens=500)

# Chat input
query = st.chat_input("Ask a question:")
if query:
    # System prompt
    system_prompt = (
        "You are an assistant for question answering tasks. "
        "Use the selected contexts of the retrieved context to answer the question. "
        "If you don't know the answer to the question, you can say 'I don't know'. "
        "Use 3 sentences maximum to keep the answer concise.\n\n"
        "{context}"
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    # Create document chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Create retrieval chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Retrieve and generate response
    result = rag_chain.invoke({"input": query})
    st.write(result["answer"])

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings, OpenAI
# import streamlit as st
# from dotenv import load_dotenv
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# import os
# from pytube import YouTube
# import re

# # Load environment variables
# load_dotenv()

# # Function to extract and clean YouTube transcript
# def get_youtube_transcript(video_url):
#     try:
#         yt = YouTube(video_url)
#         caption = yt.captions.get_by_language_code('en')  # Get English captions
#         if caption:
#             srt_text = caption.generate_srt_captions()
#             # Clean the transcript
#             text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', srt_text)
#             text = re.sub(r'\d+', '', text)  # Remove line numbers
#             text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a single space
#             text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
#             return text.strip()
#         else:
#             return None
#     except Exception as e:
#         st.error(f"Error extracting transcript: {e}")
#         return None

# # Streamlit app title
# st.title("RAG Application with YouTube Content")

# # User input for YouTube URL
# video_url = st.text_input("Enter YouTube URL:", "https://www.youtube.com/watch?v=dGby9BH9bMc")

# # Extract and process transcript
# if video_url:
#     transcript = get_youtube_transcript(video_url)
#     if transcript:
#         st.write("Transcript extracted successfully!")
#         # Split transcript into chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
#         split_data = text_splitter.split_text(transcript)

#         # Embedding and storing in Chroma
#         vectorstore = Chroma.from_texts(split_data, OpenAIEmbeddings())

#         # Retriever setup
#         retriever = vectorstore.as_retriever(search_type="cosine_similarity", search_kwargs={"n_results": 5})

#         # LLM setup
#         llm = OpenAI(temperature=0.5, max_tokens=500)

#         # Chat input
#         query = st.chat_input("Ask a question about the video:")
#         if query:
#             # System prompt
#             system_prompt = (
#                 "You are an assistant for question answering tasks. "
#                 "Use the selected contexts of the retrieved context to answer the question. "
#                 "If you don't know the answer to the question, you can say 'I don't know'. "
#                 "Use 3 sentences maximum to keep the answer concise.\n\n"
#                 "{context}"
#             )

#             # Prompt template
#             prompt = ChatPromptTemplate.from_messages(
#                 [
#                     ("system", system_prompt),
#                     ("human", "{input}")
#                 ]
#             )

#             # Create document chain
#             question_answer_chain = create_stuff_documents_chain(llm, prompt)

#             # Create retrieval chain
#             rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#             # Retrieve and generate response
#             result = rag_chain.invoke({"input": query})
#             st.write(result["answer"])
#     else:
#         st.error("No transcript available for this video.")