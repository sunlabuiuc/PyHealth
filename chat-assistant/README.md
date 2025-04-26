## LLM for PyHealth
The current LLM for pyhealth interface is deployed here: http://35.208.88.194:7861/.

### Step 1:
Merge all pyhealth related information (code and doc txt) into `pyhealth.txt`
- currently, we use https://github.com/mpoon/gpt-repository-loader

### Step 2:
Run the `ingest.py` to transform the `pyhealth.txt` into the FAISS vector database
```python
python ingest.py
```

### Step 3:
Run the retrieval augmented generation (RAG) app for Q & A based on the `pyhealth.txt` document.
```python
python app_rag.py
```


### Launch in Docker

1. Modfiy environment variables (OPENAI_API_KEY, server address...)in `Dockerfile`.
2. Build image by `docker build -t chat-pyhealth .`.
3. Debug a container by `docker run -p 0.0.0.0:7861:7861 --name chat-pyhealth-c -v ./logs/:/app/logs/ chat-pyhealth`.
4. Run a container by `docker run -d -p 0.0.0.0:7861:7861 --name chat-pyhealth-c -v ./logs/:/app/logs/ chat-pyhealth`.

```shell
## build container
docker run -d -p [host address and port]:[container port] --name [name] -v [host path]:[container path] [image]

# -d: detached
# -p: port
# --name: container name
# -v: mount directory of container to local host path

## check
docker ps
docker images

## remove
docker stop / restart chat-pyhealth-c # container
docker rm chat-pyhealth-c # container
docker rmi chat-pyhealth # image

## modify directly
docker cp [local file in host] chat-pyhealth-c:[container path]
```

## Using a Local Model 
You can also run an instance of a chat assistant that doesn't use an external LLM. There is a simple chat interface (source code is in [chat.py](/chat-assistant/chat.py)). This chat interface uses Streamlit, and the model is served using Ollama, so please install those too. 

You can find the documentation for Ollama [here](https://ollama.com). Follow the installation steps listed there.

To install Streamlit, run:

```{python}
pip install streamlit
```

### Steps to Run
Please follow the steps (in order!) to get started.

1. Navigate into the `chat-assistant` directory
2. Generate the documents needed for embeddings:
```{python}
python generate_context.py
```
3. Start the app:
```{python}
streamlit run chat.py
```

**Let me know if you want to join and help us improve the current interface.**
