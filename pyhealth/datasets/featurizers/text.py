from transformers import AutoTokenizer, AutoModel
import torch


class TextFeaturizer:

    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, value):
        # Tokenize and process the input text
        inputs = self.tokenizer(value, return_tensors="pt", truncation=True,
                                padding=True)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use the CLS token embedding as the representation (can also use pooling for sentence embeddings)
        embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding

        return embedding


if __name__ == "__main__":
    sample_text = "This is a sample text input for the TextFeaturizer."
    featurizer = TextFeaturizer()
    print(featurizer)
    print(type(featurizer))
    print(featurizer.encode(sample_text))
