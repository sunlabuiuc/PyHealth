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

**Let me know if you want to join and help us improve the current interface.**
