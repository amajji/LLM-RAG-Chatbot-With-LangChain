############################################################################################

#                                  Author: Anass MAJJI                                     #

#                               File Name: streamlit_app.py                                #

#                           Creation Date: May 06, 2024                                    #

#                         Source Language: Python                                          #

#         Repository:    https://github.com/amajji/LLM-RAG-Chatbot-With-LangChain          #

#                              --- Code Description ---                                    #

#    Deploy LLM RAG Chatbot with Langchain on a Streamlit web application using only CPU   #

############################################################################################


############################################################################################
#                                   Packages                                               #
############################################################################################


# Import Python Libraries
import streamlit as st
import seaborn as sns
import pathlib
import folium
import os

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# from langchain.llms import CTransformers
from langchain_community.llms import CTransformers
import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.utils import DistanceStrategy


#########################################################################################
#                                Variables                                              #
#########################################################################################

st.set_page_config(layout="wide")
STREAMLIT_STATIC_PATH = str(pathlib.Path(st.__path__[0]) / "AI_Hackathon_Dataset/pdf")
STREAMLIT_STATIC_PATH = "./AI_Hackathon_Dataset/pdf"


#########################################################################################
#                                Functions                                              #
#########################################################################################


# @st.cache
def create_vector_db(data_path):

    """function to create vector db provided the pdf files"""

    # define the docs's path
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)

    # load documents
    documents = loader.load()

    # use recursive splitter to split each document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # generate embeddings for each chunk
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        multi_process=True,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"},
    )

    # create the vector database
    db = FAISS.from_documents(texts, embeddings)

    return db


# @st.cache
def load_llm(temperature, max_new_tokens, top_p, top_k):
    """Load the LLM model"""

    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    # return the LLM
    return llm


@st.cache_resource()
def q_a_llm_model(temperature, max_new_tokens, top_p, top_k):
    """
    This function loads the LLM model, gets the relevent
    docs for a given query and provides an answer
    """

    # create the vector database
    vector_db = create_vector_db(STREAMLIT_STATIC_PATH)

    # get the top_k relevent documents
    # print(f"\nStarting retrieval for {user_query=}...")
    # retrieved_docs = vector_db.similarity_search(query=user_query, k=5)
    # print(
    #    "\n==================================Top document=================================="
    # )
    # print(retrieved_docs[0].page_content)
    # print("==================================Metadata==================================")
    # print(retrieved_docs[0].metadata)

    # load the model
    llm_model = load_llm(temperature, max_new_tokens, top_p, top_k)

    # Create a retriever object from the 'db' with a search configuration where
    # it retrieves up to top_k relevant splits/documents.
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})

    # Create a question-answering instance (qa) using the RetrievalQA class.
    # It's configured with a language model (llm), a chain type "refine,"
    # the retriever we created, and an option to not return source documents.
    q_a = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="refine",
        retriever=retriever,
        return_source_documents=False,
    )

    return q_a


#########################################################################################
#                                Main code                                              #
#########################################################################################


# First streamlit's page
def page_1():

    # define the title
    st.title("✨ LLM with RAG")

    # quick decription of the webapp
    st.markdown(
        """
        This interactive dashboard is designed to extract any information from external documents. 
        The LLM used is LLaMA2-7B with LangChain for RAG. The user has the possibility to ask questions and the LLM provides 
        an appropriate answer from the available documents.
        To speed the inference time, we've used the quantized version of the model with **GGML** quantization approach, 
        it can run on only **CPU** processors.
        """
    )

    st.markdown(
        """
        Below, a chat to interact with the LLM.
        """
    )

    # Text generation params
    st.subheader("Text generation parameters")
    temperature = st.sidebar.slider(
        "Temperature", min_value=0.1, max_value=1.0, value=0.1, step=0.01
    )
    top_p = st.sidebar.slider(
        "Top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01
    )
    top_k = st.sidebar.slider("Top_p", min_value=0, max_value=100, value=20, step=10)
    max_length = st.sidebar.slider(
        "Max_length", min_value=64, max_value=4096, value=512, step=8
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.caption("🚀 A Streamlit chatbot powered by OpenAI")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Create a question-answering instance with the provided params
        q_a = q_a_llm_model(temperature, max_length, top_p, top_k)

        # get the result
        result = q_a.run({"query": prompt})
        # print("------------ : ", result)

        st.session_state.messages.append(
            {"role": "assistant", "content": "voici le message de retour"}
        )
        st.chat_message("assistant").write(result)


def main():

    """A streamlit app template"""
    st.sidebar.title("Menu")

    PAGES = {
        "🎈 LLaMA2-7B LLM": page_1,
    }

    # Select pages
    # Use dropdown if you prefer
    selection = st.sidebar.radio("Select your page : ", list(PAGES.keys()))
    sidebar_caption()

    PAGES[selection]()

    st.sidebar.title("About")
    st.sidebar.info(
        """
    Web App URL: <https://amajji-streamlit-dash-streamlit-app-8i3jn9.streamlit.app/>
    GitHub repository: <https://github.com/giswqs/streamlit-geospatial>
    """
    )

    st.sidebar.title("Contact")
    st.sidebar.info(
        """
    MAJJI Anass 
    [GitHub](https://github.com/amajji) | [LinkedIn](https://fr.linkedin.com/in/anass-majji-729773157)
    """
    )


if __name__ == "__main__":

    main()
