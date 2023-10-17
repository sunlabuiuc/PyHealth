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


**Let me know if you want to join and help us improve the current interface.**
