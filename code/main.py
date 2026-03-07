from typing import Union
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize

from schemas import TextRank, JustText

SENTIMENT_MODEL_PATH = Path("sentiment_classifier.pkl")
sentiments = ["negative", "neutral", "positive"]

SPAM_MODEL_PATH = Path("spam_classifier.pkl")

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)  
    spam_model = joblib.load(SPAM_MODEL_PATH)

    # attach it to app.state so endpoints can use it
    app.state.sentiment_model = sentiment_model
    app.state.spam_model = spam_model

    print("Models loaded.")
    yield

    # Shutdown logic here if needed
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

@app.post("/sentiment/")
def read_text_sentiment(request: JustText):
    """
    Analyze the sentiment of a given text string.

    Parameters:
        text (str): The input text to analyze.

    Returns:
        dict: {
            "sentiment": "positive" | "negative" | "neutral",
            "confidence": float  # between 0 and 1
        }

    Raises:
        400: No input text provided.
        503: Model is not loaded and ready.

    NOTE: the API will return an error if the request is not a valid JSON.
    """
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="No input text provided.")

    if app.state.sentiment_model is None:
        raise HTTPException(status_code=503, detail="Model failed to load.")
    
    print("Checking sentiment for: ", text)

    # Checks which sentiment belongs to the text
    score = app.state.sentiment_model.decision_function([text])[0]
    pred = 1 / (1 + np.exp(-score))

    # Calculates the confidence 
    confidence = pred.max()
    response = {
        "sentiment": sentiments[pred.argmax()],
        "confidence": round(confidence, 3)
    }
    return response

@app.post("/spam/")
def read_text_spam(request: JustText):
    """
    Analyze whether a given text is spam or not.

    Parameters:
        text (str): The input text to analyze.

    Returns:
        dict: {
            "is_spam": True | False,
            "confidence": float  # between 0 and 1
        }

    Raises:
        400: No input text provided.
        503: Model is not loaded and ready.

    NOTE: the API will return an error if the request is not a valid JSON.
    """
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="No input text provided.")

    if app.state.spam_model is None:
        raise HTTPException(status_code=503, detail="Model failed to load.")
    
    print("Checking spam for:", text)

    # Checks if the input text is spam
    spam = True if app.state.spam_model.predict([text])[0] == "spam" else False

    # Retrieves the confindence of the prediction
    confidence = max(app.state.spam_model.predict_proba([text])[0])

    response = {
        "is_spam": spam,
        "confidence": round(confidence, 3)
    }
    return response

@app.post("/textrank/")
def read_text_textrank(request: TextRank):
    """
    Summarizes long texts, extracting n most important sentences and returning it in the original order.

    Parameters:
        text (str): The input text to summarize.
        n (int): The number of sentences to be returned from the original text (optional).

    Returns:
        dict: {
            "summary": str
        }

    Raises:
        400: No input text provided.
    
    NOTE: the API will return an error if the request is not a valid JSON.
    """
    text = request.text.replace("\n", " ").replace("\r", " ").strip()
    n = request.n
    if not text:
        raise HTTPException(status_code=400, detail="No input text provided.")
    
    print("Shortening this text:", text)

    sentences = sent_tokenize(text)

    # Tokenization and vectorization
    vectorizer = TfidfVectorizer(stop_words="english")
    vec = vectorizer.fit_transform(sentences)

    # Calculating similarity between the sentences 
    similarity_matrix = cosine_similarity(vec)

    # Transforming the similarity matrix to undirected graph
    nx_graph = nx.from_numpy_array(similarity_matrix)

    # Calculating scores from the graph
    scores = nx.pagerank(nx_graph)

    # Keep the indices of the sentences with the highest scores
    top_indices = sorted(scores, key=scores.get, reverse=True)[:n] if n <= len(sentences) else sorted(scores, key=scores.get, reverse=True)
    
    # Sort the indices, so that sentences appear in original order from text
    top_indices.sort()
    
    # Return the most n most important sentences 
    summary = " ".join([sentences[i] for i in top_indices])

    response = {
        "summary": summary
    }
    return response
