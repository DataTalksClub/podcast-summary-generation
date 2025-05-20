from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat

def evaluate_summary(original, summary):
    vectorizer = TfidfVectorizer().fit_transform([original, summary])
    sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
    return {
        "Compression Ratio": round(len(summary) / len(original), 3),
        "Coverage Score": round(sim, 3),
        "Readability": textstat.flesch_reading_ease(summary)
    }
