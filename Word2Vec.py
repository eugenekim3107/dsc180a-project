from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import json
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

class word2vec:
    def __init__(self, X, tokenizer, vector_size=100, window=5, min_count=2, sg=1, sample=1e-3, workers=4):
        self.X_tokenized = [tokenizer(doc.lower()) for doc in X]
        self.model = Word2Vec(sentences=self.X_tokenized, vector_size=vector_size, window=window, min_count=min_count, workers=workers,sg=sg, sample=sample)
        
    def classify(self, seed_words):
        label_representations = {}
        for label, seeds in seed_words.items():
            label_representations[label] = np.mean([self.model.wv[word] for word in seeds if word in self.model.wv], axis=0)

        document_labels = []

        for doc in self.X_tokenized:
            doc_representation = np.mean([self.model.wv[word] for word in doc if word in self.model.wv], axis=0)
            similarities = {label: cosine_similarity([doc_representation], [label_vec])[0][0] for label, label_vec in label_representations.items()}
            document_labels.append(max(similarities, key=similarities.get))
        
        return document_labels
    
def main():
    
    # 20 Newsgroup Dataset (Coarse)
    with open("data/20news/coarse/df.pkl", "rb") as file:
        data = pickle.load(file)
    with open("data/20news/coarse/seedwords.json", "rb") as file:
        seedwords = json.load(file)

    X = data["sentence"]
    y = data["label"]

    tokenizer = word_tokenize
    classifier = word2vec(X, tokenizer, vector_size=100, window=10, min_count=2)

    pred = classifier.classify(seedwords)

    f1_macro = f1_score(y,pred,average='macro')
    f1_micro = f1_score(y,pred,average='micro')
    print("20 Newsgroup Dataset (Coarse)")
    print(f"F1_score (macro): {f1_macro}")
    print(f"F1_score (micro): {f1_micro}")

    # 20 Newsgroup Dataset (Fine)
    with open("data/20news/fine/df.pkl", "rb") as file:
        data = pickle.load(file)
    with open("data/20news/fine/seedwords.json", "rb") as file:
        seedwords = json.load(file)

    X = data["sentence"]
    y = data["label"]

    tokenizer = word_tokenize
    classifier = word2vec(X, tokenizer, vector_size=350, window=20, min_count=3, workers=15, sg=0)

    pred = classifier.classify(seedwords)

    f1_macro = f1_score(y,pred,average='macro')
    f1_micro = f1_score(y,pred,average='micro')
    print("20 Newsgroup Dataset (Fine)")
    print(f"F1_score (macro): {f1_macro}")
    print(f"F1_score (micro): {f1_micro}")

    # NYT Dataset (Coarse)
    with open("data/nyt/coarse/df.pkl", "rb") as file:
        data = pickle.load(file)
    with open("data/nyt/coarse/seedwords.json", "rb") as file:
        seedwords = json.load(file)

    X = data["sentence"]
    y = data["label"]

    tokenizer = word_tokenize
    classifier = word2vec(X, tokenizer, vector_size=350, window=20, min_count=3, workers=15, sg=0)

    pred = classifier.classify(seedwords)

    f1_macro = f1_score(y,pred,average='macro')
    f1_micro = f1_score(y,pred,average='micro')
    print("NYT Dataset (Coarse)")
    print(f"F1_score (macro): {f1_macro}")
    print(f"F1_score (micro): {f1_micro}")

    # NYT Dataset (Fine)
    with open("data/nyt/fine/df.pkl", "rb") as file:
        data = pickle.load(file)
    with open("data/nyt/fine/seedwords.json", "rb") as file:
        seedwords = json.load(file)

    X = data["sentence"]
    y = data["label"]

    tokenizer = word_tokenize
    classifier = word2vec(X, tokenizer, vector_size=350, window=20, min_count=3, workers=15, sg=0)

    pred = classifier.classify(seedwords)

    f1_macro = f1_score(y,pred,average='macro')
    f1_micro = f1_score(y,pred,average='micro')
    print("NYT Dataset (Fine)")
    print(f"F1_score (macro): {f1_macro}")
    print(f"F1_score (micro): {f1_micro}")


if __name__ == "__main__":
    main()