def embedding_retrieve(model, tokenizer, phrase):
    # Encode the sentence
    inputs = tokenizer(phrase, return_tensors='pt')

    # Get the model's output 
    outputs = model(**inputs)

    # Extract the embeddings
    embedding = outputs.last_hidden_state.mean(dim=1)

    # Now, `embedding` is a tensor that contains the embedding for your sentence.
    # You can convert it to a numpy array if needed:
    embedding = embedding.detach().numpy().tolist()[0]

    return embedding
