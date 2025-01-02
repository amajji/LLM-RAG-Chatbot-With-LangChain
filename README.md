# Deployment on AWS of a Q&A LLM Chatbot using only CPU
Data scientist | [Anass MAJJI](https://www.linkedin.com/in/anass-majji-729773157/)

***


## :monocle_face: Description

<p align="center">
 <img src="images/aws2.png" width="50%" />
</p>

- In this project, we deploy on **AWS** a **LLM RAG Chatbot** with **Langchain** on a **Streamlit** web application using only **CPU**. </br>
The LLM model aims at extracting relevent informations from external documents. In our case, we've used the quantized version of **Llama-2-7B** with **GGML** quantization approach, it can be used with only **CPU** processors.

- Traditionally, the LLM has only relied on prompt and the training data on which the model was trained. However, this approach posed limitations in terms of knowledge especially when dealing with large datasets that exceed token length constraints. To address this challenge, RAG (Retrieval Augmented Generation) intervenes by enriching the LLM with new and external data sources.

Before making a demo of the streamlit web application, let's walk through the details of the RAG approach to understand how it works. The **retriever** acts like an internal search engine : given a user query, it returns a few relevent elements from the external data sources. Here are the main steps of the RAG system : 

- **1** - Split each document of our knowledge into chunks and get their embeddings: we should keep in mind that when embbeding documents, we will use a model that accepts a certain maximum sequence length max_seq_length. 

- **2** - Once all the chunks are embedded, we store them in a vector database. When the user types a query, it gets embedded by the same model previously used, then a similarity search returns the top_k closest chunks from the vector database. To do so, we need two elements : 1) a metric to mesure the distance between emdeddings (Euclidean distance, Cosinus similarity, Dot product) and 2) a search algorithm to find the closest elements (Facebook's FAISS). Our particular model works well with cosinus similarity.

- **3** - Finally, the content of the retrieved documents is aggregated together into the "context", which is also aggregated with the query into the "prompt". It's then fed to the LLM to generate answers.

- Below a perfect illustration of the RAG steps : 

 
<p align="center">
 <img src="images/RAG_workflow.png" width="50%" />
</p>



In order to reach a good accuracy with the LLMs, we need to better understand and choose each hyperparameter. Before deeping dive into the details, let's remind the LLM's decoding process. As we know, LLMs rely on transformers, each one is composed with two main blocs : **encoder** which converts the input tokens into embeddings i.e numerical values and **decoder** which tries to generate tokens from embeddings (the opposit of the encoder). There are two main types of decoding : **greedy** and **sampling**. With greedy decoding, the model simply chooses the token with the highest probability at each step during inference.

 
With sampling decoding, in contrast, the model select a subset of potential output tokens and select randomly one of them to add to the output text. This creates more variability and helps the LLM to be more creative. However, opting for sampling decoder increases the risk of incorrect responses.

 
When opting for sampling decoding, we have two additional hyperparameters which impact the performance of the model : Top_k and Top_p.


- **top_k** : The top_k hyperparameter is an integer that ranges from **1** to **100**. It represents the k tokens with the highest probabilities. To more understand the idea behind, let's take an example :  we have this sentence "I went to meet a friend" and we want to predict the next token, we have 3 possiblities 1) in the center of the city 2) to eat together 3) on the other side of town. Now, let assume that "in", "to" and "on" have respectively the following probabilities [0.23, 0.12, 0.30]. With top_k = 2, we are going to select only two tokens with the highest probabilities:  "in" and "on" in our case. Then the model chooses randomly one of them.

- **top_p** : is a decimal feature that ranges from **0.0** to **1.0**. The model try to choose a subset of tokens with their cumulative probabilities equals to top_p value. Considering the above example, with a top_p = 0.55, the only tokens with their cumulative probabilities inferior to 0.55 are "in" and "on".

 
- **temperature**: performs a similar function as the above top_k and top_p hyperparameters. It ranges from **0** to **2** (maximum of creativity). The idea behind is to change the probability distribution of the output tokens. With a lower temperature value, the model amplifies the probabilities, means tokens with higher probabilities become even more likely to be output and vice-versa. The lower values are used when we want to generate predictable responses.
In contrast, higher values cause convergence of the probabilities : they become close to each other. Using them push the LLM to be more creative.


Another paramater we should take into consideration is the memory needed to run the LLM: for a model with N parameter and a full precision (fp32) the memory needed is N x 4Bytes. However, when we use quantization, we divide by (4 Bytes/ new precision). With fp16, the new memory is divided by 4 Bytes/ 2 Bytes. 


## :rocket: Repository Structure

The repository contains the following files & directories:
- **app** : it contains the streamlit code for the **LLM RAG Chatbot** webapp.
- **Dockerfile** : it contains the instructions to build the docker image. 
- **images** : this folder contains all images used on the README file.
- **requirements.txt:** all the packages used in this project.

 

 

## :chart_with_upwards_trend: Demontration

In this section, we are going to make a demonstration of the streamlit webapp. The user can ask any question and the chatbot will answer. 

To launch the deployment of the streamlit app with docker, type the following commands :

- docker build -t streamlit . : to build the docker image

- docker run -p 8501:8501 streamlit: to launch the container based on our image

To view our app, users can browse to http://0.0.0.0:8501 or http://localhost:8501

## :fire: Deployment on AWS using Github actions and Github Container Registry

If you are interested in deploying the LLM web application on AWS. Below a step-by-step guide to follow :

**Step 1**: Push Your Docker Image to GitHub Container Registry (you can also use **Amazon Elastic Container Registry (ECR)**):
- docker build -t ghcr.io/<your-username>/<your-repo-name>:latest . : To make sure that the Dockerfile is correctly set up to run the streamlit Q&A model. 
- echo $CR_PAT | docker login ghcr.io -u <your-username> --password-stdin : Log in to GitHub Container Registry ($CR_PAT is your GitHub Personal Access Token, which should have write:packages, read:packages, and delete:packages scope.)
- docker push ghcr.io/<your-username>/<your-repo-name>:latest : Tag and push the image to GitHub Container Registry

**Step 2**: Set Up AWS EKS (Elastic Kubernetes Service):
- After creating an AWS account, you need to create an IAM user. To fully manage the EKS container, you should attach the following policies to your profil : AmazonEKSClusterPolicy, AmazonEKSServicePolicy, AmazonEKSWorkerNodePolicy, AmazonEC2ContainerRegistryReadOnly, AmazonEC2FullAccess, AmazonVPCFullAccess, IAMFullAccess, AmazonCloudFormationFullAccess, ElasticLoadBalancingFullAccess.

- Once the IAM user has the necessary permissions and has been correctly created, you will see a success screen with the user’s Access Key ID and Secret Access Key. You need then to configure AWS CLI with the new user's Credentials using **aws configure** command. 

**Step 3**: Set Up AWS EKS (Elastic Kubernetes Service):
- eksctl create cluster --name llama-cluster --region <region> --nodes 2 --node-type t3.medium --managed : This command creates on AWS a Kubernetes cluster named llama-cluster with 2 nodes of type t2.medium.
- aws eks --region <region> update-kubeconfig --name llama-cluster : After creating the cluster, we run the following command to update kubectl to use the newly created EKS cluster.

**Step 4**: Create Kubernetes Deployment YAML:
Create a llama-deployment.yaml to define your Kubernetes deployment for the Streamlit app. This will include details like container image, resources (CPU, memory), environment variables, etc.

**Step 5**: Set Up GitHub Actions for CI/CD:
In your GitHub repo, create the .github/workflows/ci-cd.yaml file. This will contain the steps for building the Docker image, pushing it to GitHub Container Registry, and deploying it to AWS EKS.

**Step 6**: Apply Kubernetes Deployment:
Once the GitHub Action is triggered (on push to the main branch), the deployment will be applied to the EKS cluster automatically:
- kubectl apply -f llama-deployment.yaml
- kubectl apply -f llam-service.yaml

**Step 7** : Expose the Application (Service):
Once the deployment is successful, expose the application using a LoadBalancer. Kubernetes will automatically provision an AWS ELB (Elastic Load Balancer).

You can check the service’s external IP after it's created by running:
- kubectl get services

Once the LoadBalancer is up and running, access the Streamlit app via the EXTERNAL-IP provided by the service.

In order to monitor and maintain the deployment, you can use the following commands:
- kubectl get pods : check the status of your pods
- kubectl logs <pod-name>: check the logs of the pod-name pod
- kubectl scale deployment llama-deployment --replicas=3 : Sclale the deployment if needed.

## :chart_with_upwards_trend: Performance & results

---

## :mailbox_closed: Contact
For any information, feedback or questions, please [contact me][anass-email]


[anass-email]: mailto:anassmajji34@gmail.com