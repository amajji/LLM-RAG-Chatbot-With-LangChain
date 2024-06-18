# LLM RAG Chatbot.
Data scientist | [Anass MAJJI](https://www.linkedin.com/in/anass-majji-729773157/)

***


## :monocle_face: Description

- In this project, we deploy an **LLM RAG Chatbot** with **Langchain** on a **Streamlit** app. </br>
The development of this LLM model aims at extracting relevent informations from external documents. Traditionally, the LLM has only relied on prompt and
the training data on which the model was trained. However, this approach posed limitations in terms of knowledge especially when dealing with
large datasets that exceed token length constraints. To address this challenge, RAG (Retrieval Augmented Generation) intervenes by enriching the LLM with new and external data sources.


 
<p align="center">
 <img src="images/RAG_workflow.png" width="50%" />
</p>



## :rocket: Repository Structure

The repository contains the following files & directories:
- **app** : it contains the streamlit code for the **LLM RAG Chatbot** webapp.
- **images** : this folder contains all images used on the README file.
- **requirements.txt:** all the packages used in this project.

 

 

## :chart_with_upwards_trend: Demontration

In this section, we are going to make a demonstration of the streamlit webapp. The user can ask the chatbot  
To launch the deployment of the streamlit app with docker, type the following commands :

- docker build -t webapp_ocr . : to build the docker image

- docker run -p 8000:80 webapp_ocr:latest : to launch the container based on our image

 
To see the streamlit, clic on .

 

## :chart_with_upwards_trend: Performance & results

---

## :mailbox_closed: Contact
For any information, feedback or questions, please [contact me][anass-email]


[anass-email]: mailto:anassmajji34@gmail.com