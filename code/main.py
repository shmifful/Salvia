import os
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi import FastAPI, HTTPException, Security, Depends, BackgroundTasks, Request
from contextlib import asynccontextmanager
import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize
from fastapi.security import APIKeyHeader
from schemas import TextRank, JustText
import time

load_dotenv()

SENTIMENT_MODEL_PATH = Path("sentiment_classifier.pkl")
sentiments = ["negative", "neutral", "positive"]

SPAM_MODEL_PATH = Path("spam_classifier.pkl")

# Get supabase creds
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_ROLE")
supabase: Client = create_client(url, key)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Verify key
async def validate_api_key(api_key: str = Security(api_key_header)):
    # TODO: Don't let users regenerate key to reset their usage
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key"
        )
    
    # Checks if the API key exists in the DB
    response = (
        supabase.table("api_keys")
        .select("user_id")
        .eq("api_key", api_key)
        .execute()
    )

    if not response.data:
        raise HTTPException(
            status_code=403, 
            detail="Invalid API key"
        )
    
    return api_key

# Write data in DB
async def track_usage(api_key, endpoint, status, data=None, res=None, latency=0):
    response = (supabase
        .table("usage")
        .insert({
            "api_key": api_key,
            "endpoint": endpoint,
            "data": data,
            "response": res,
            "status": status,
            "latency": latency
        })
        .execute()
    )
    
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

@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    start_time = time.perf_counter()
    
    # Process the request
    response = await call_next(request)
    
    # Calculate latency
    latency = int((time.perf_counter() - start_time) * 1000)
    
    # Get the API Key from headers manually
    api_key = request.headers.get("X-API-Key")
    
    if api_key and response.status_code != 200:
        await track_usage(
            api_key=api_key, 
            endpoint=request.url.path, 
            status=response.status_code, 
            latency=latency
        )
        
    return response

@app.post("/nlp/sentiment")
async def read_text_sentiment(
    request: JustText,
    bgTask: BackgroundTasks,
    key: dict = Depends(validate_api_key),
    ):
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
    start_time = time.perf_counter()

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

    end_time = time.perf_counter()
    latency = int((end_time - start_time) * 1000)

    response = {
        "sentiment": sentiments[pred.argmax()],
        "confidence": round(confidence, 3)
    }

    bgTask.add_task(track_usage, 
                    api_key=key, 
                    endpoint="/nlp/sentiment",
                    data={"text": text},
                    res=response,
                    status=200,
                    latency=latency
                    )

    return response

@app.post("/nlp/spam")
def read_text_spam(
    request: JustText,
    bgTask: BackgroundTasks,
    key: dict = Depends(validate_api_key)
    ):
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
    start_time = time.perf_counter()

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

    end_time = time.perf_counter()
    latency = int((end_time - start_time) * 1000)

    response = {
        "is_spam": spam,
        "confidence": round(confidence, 3)
    }

    bgTask.add_task(track_usage, 
                    api_key=key, 
                    endpoint="/nlp/spam",
                    data={"text": text},
                    res=response,
                    status=200,
                    latency=latency
                    )
    
    return response

@app.post("/nlp/textrank")
def read_text_textrank(
    request: TextRank,
    bgTask: BackgroundTasks,
    key: dict = Depends(validate_api_key)
    ):
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
    start_time = time.perf_counter()

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

    end_time = time.perf_counter()
    latency = int((end_time - start_time) * 1000)

    response = {
        "summary": summary
    }

    bgTask.add_task(track_usage, 
                    api_key=key, 
                    endpoint="/nlp/text",
                    data={
                        "text": text,
                        "n": n
                    },
                    res=response,
                    status=200,
                    latency=latency
                    )
    return response
