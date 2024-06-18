# LLM RAG Chatbot (only CPU)
Data scientist | [Anass MAJJI](https://www.linkedin.com/in/anass-majji-729773157/)

***


## :monocle_face: Description

- In this project, we deploy a **LLM RAG Chatbot** with **Langchain** on a **Streamlit** web application using only **CPU**. </br>
The LLM model aims at extracting relevent informations from external documents. In our case, we've used the quantized version of **Llama-2-7B** with **GGML** quantization approach, it can be used with only **CPU** processors.

- Traditionally, the LLM has only relied on prompt and the training data on which the model was trained. However, this approach posed limitations in terms of knowledge especially when dealing with large datasets that exceed token length constraints. To address this challenge, RAG (Retrieval Augmented Generation) intervenes by enriching the LLM with new and external data sources.

Before making a demo of the streamlit web application, let's walk through the details of the RAG approach to understand how it works. The **retriever** acts like an internal search engine : given a user query, it returns a few relevent elements from the external data sources. Here are the main steps of the RAG system : 

- **1** - Split each document of our knowledge into chunks and get their embeddings: we should keep in mind that when embbeding documents, we will use a model that accepts a certain maximum sequence length max_seq_length. 

- **2** - Once all the chunks are embedded, we store them in a vector database. When the user types a query, it gets embedded by the same model previously used, then a similarity search returns the top_k closest chunks from the vector database. To do so, we need two elements : 1) a metric to mesure the distance between emdeddings (Euclidean distance, Cosinus similarity, Dot product) and 2) a search algorithm to find the closest elements (Facebook's FAISS). Our particular model works well with cosinus similarity.

- **3** - Finally, the content of the retrieved documents is aggregated together into the "context", which is also aggregated with the query into the prompt. It's then fed to the LLM to generate answers.

- Below a perfect illustration of the RAG steps : 

 
<p align="center">
 <img src="images/RAG_workflow.png" width="50%" />
</p>



## :rocket: Repository Structure

The repository contains the following files & directories:
- **app** : it contains the streamlit code for the **LLM RAG Chatbot** webapp.
- **Dockerfile** : it contains the instructions to build the docker image. 
- **images** : this folder contains all images used on the README file.
- **requirements.txt:** all the packages used in this project.

 

 

## :chart_with_upwards_trend: Demontration

In this section, we are going to make a demonstration of the streamlit webapp. The user can ask any question and the chatbot will answer based on the elements   

To launch the deployment of the streamlit app with docker, type the following commands :

- docker build -t streamlit . : to build the docker image

- docker run -p 8501:8501 streamlit: to launch the container based on our image


To view our app, users can browse to http://0.0.0.0:8501 or http://localhost:8501

 

## :chart_with_upwards_trend: Performance & results

---

## :mailbox_closed: Contact
For any information, feedback or questions, please [contact me][anass-email]


[anass-email]: mailto:anassmajji34@gmail.com