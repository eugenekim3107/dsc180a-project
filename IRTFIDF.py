import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from sklearn.metrics import f1_score

class IRTFIDF:
    def __init__(self, X, vectorizer):
        self.X = X
        self.N = len(X)
        self.vectorizer = vectorizer
        self.X_tfidf = vectorizer.fit_transform(X)
    
    def compute_tfidf(self, X_idx, words):
        score = 0
        for w in words:
            if w not in self.vectorizer.vocabulary_:
                score += 0
            else:
                word_index = self.vectorizer.vocabulary_[w]
                score += self.X_tfidf[X_idx, word_index]
        return score
        
    def classify(self, seedwords):
        documents_labels = []
        for i in range(self.X.shape[0]):
            scores = {}
            for label, words in seedwords.items():
                all_words = words + [label]
                scores[label] = self.compute_tfidf(i, all_words)
            if sum(scores.values()) == 0:
                documents_labels.append("sci")
            else:
                documents_labels.append(max(scores, key=scores.get))
        return documents_labels
    
def main():

    # 20 Newsgroup Dataset (Coarse)
    with open("data/20news/coarse/df.pkl", "rb") as file:
        data = pickle.load(file)
    with open("data/20news/coarse/seedwords.json", "rb") as file:
        seedwords = json.load(file)

    X = data["sentence"]
    y = data["label"]

    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    classifier = IRTFIDF(X, vectorizer)
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

    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    classifier = IRTFIDF(X, vectorizer)

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

    vectorizer = TfidfVectorizer(stop_words="english")
    classifier = IRTFIDF(X, vectorizer)

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

    vectorizer = TfidfVectorizer(stop_words="english")
    classifier = IRTFIDF(X, vectorizer)

    pred = classifier.classify(seedwords)

    f1_macro = f1_score(y,pred,average='macro')
    f1_micro = f1_score(y,pred,average='micro')
    print("NYT Dataset (Fine)")
    print(f"F1_score (macro): {f1_macro}")
    print(f"F1_score (micro): {f1_micro}")

if __name__ == "__main__":
    main()