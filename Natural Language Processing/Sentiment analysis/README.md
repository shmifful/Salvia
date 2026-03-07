<details>
    <summary><b><u>Table of Contents (Click to expand)</u></b></summary>

  - [How to use](#how-to-use)
  - [Model structure](#model-structure)
  - [Use cases](#use-cases)
</details>

# How to use 
## Sentiment Analysis endpoint
```
https://api.salvia.dev/sentiment/
```

## Input
```python
text: str
```

## Output
```python
dict : {
    "sentiment": "positive" | "neutral" | "negative",
    "confidence": float # range between 0 and 1
}
```

## Raises 
`400`: No input text provided. 

`503`: Model is not loaded and ready.

## Usage
You can try this endpoint yourself (and many more) at [api.salvia.dev/docs](api.salvia.dev/docs).
### `python`
```python
import requests

URL = "https://api.salvia.dev/sentiment/" 
text = "I love pizza" 
res = requests.post(URL, , json={"text": text})

print(res.json())
# Output: {'sentiment': 'positive', 'confidence': 0.956}
```
`NOTE`: the API will return an error if the request is not a valid JSON.

# Model structure
This model uses a `Linear Support Vector Machine (LinearSVC)`. Rather than just counting the words, the model uses a hybrid approach to try understand the context of the string.

## `TF-IDF Vectorization`
The model uses a hybrid approach to get different perspectives:
- Word-level ngrams (1-2): This captures the relationship between words, e.g. "not" and "good" put together change their meaning.
- Character-level ngrams (1-2): This looks at character patterns, making it more useful for casual typers, such that the model can still be effective against typos, slangs, and weird punctuation you find on Reddit posts or on YouTube comments.

## `LinearSVC`
`LinearSVC` is a high performance classifier that is able to handle high dimensional text data. In the training data, there was an imbalance in the quantity of each labels, so to counteract this, `class_weight` was set to `balanced`.

## Pros
* Scales well with large datasets.
* High accuracy.

## Cons
* Requires a lot of labelled data.
* Always requires new ingestions of data to keep it up to date (e.g., to teach the model slangs).
* Unable to handle out-of-vocabulary words.
* Struggles to handle nuances like sarcasm. 

# Use cases
This API is used to analyse various text and understand the sentiment behind the string. Where would this be useful? 

Let's say you have a product that sells thousands of units a week. For every 10 people who buy the product, 1 person leaves a review. However, there are so many reviews that it is not feasible to read all of them.

So, what do you do? Each time someone leaves a review, you can run this API, analyse their review, and determine whether the review was positive, neutral, or negative. You can then aggregate the sentiments, to get a general sense of how positively or negatively people feel about your product.

Here is a list of text that this API could analyse:
* Amazon product reviews
* YouTube comments
* Reddit posts/comments
* Instagram comments
* Google Maps reviews
* etc...