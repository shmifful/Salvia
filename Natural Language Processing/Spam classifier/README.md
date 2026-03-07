<details>
    <summary><b><u>Table of Contents (Click to expand)</u></b></summary>

  - [How to use](#how-to-use)
  - [Model structure](#model-structure)
  - [Use cases](#use-cases)
</details>

# How to use
## Spam Filter endpoint
```
https://api.salvia.dev/spam/
```

## Input
```python
text: str
```

## Output
```python
dict : {
    "is_spam": True | False,
    "confidence": float # range between 0 and 1
}
```

## Raises 
`400`: No input text provided. 

`503`: Model is not loaded and ready.

## Usage
You can try this endpoint yourself (and many more) at [api.salvia.dev/docs](api.salvia.dev/docs).
### `python`
```py
import requests

URL = "https://api.salvia.dev/spam/" 
text = "Congratulations! You have won a free iPhone. Click here to claim." 
res = requests.post(URL, , json={"text": text})

print(res.json())
# Output: {'is_spam': True, 'confidence': 0.959}
```
`NOTE`: the API will return an error if the request is not a valid JSON.


# Model structure
When searching for a dataset to use to train the model, this dataset by [UCI Machine Learning](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) was the most prominent one, as it is the most upvoted on Kaggle for spam classification and most used by other people. 

## Dataset exploration
This dataset contains two columns that are of interest for this project, which are `v1` and `v2`.

`v1` are the labels (in other words our `y`). `v2` are our features (in other words our `x`).

## Building the model
Before training the model, the features and labels are split into training data and testing data, with a ratio of 80/20 respectively. This is done such that the model is able to generalize preventing overfitting, and by the end of the training, the model can be evaluated using data that it has not seen before. Random state is set to `42`, which is an arbitrary number, so that the same model can be reproduced by someone else using the same seed.

Rather than manually pre-process the data before feeding to the model, `Pipeline` is used such that raw text data can be used as inputs, and the model will handle both the pre-processing and processing stage.

### TF-IDF Vectorization

`TfidfVectorizer` handles the pre-processing stages, which include tokenization and vectorization. Tokenization is the process of breaking down sentences down to words and subwords called "tokens" to make the data more manageable to work with. Vectorization is the process of turning the tokens generated into numerical vectors, such that it can be understood by the computer, as it does not understand raw text. 

There are some words in the English language (e.g., "the", "a", "and", etc..) which do not help us recognize whether a message is spam or not, so to ignore these words we set `stop_words='english'`.

### Logistic Regression
After pre-processing the data, this is the classifier which decides whether a given text is spam or ham.

Logistic Regression is essentially a sigmoid function, which squashes the output to a probability value between 0 and 1, rendering it useful for binary classification problems, such as in spam filtering.

The formula for the sigmoid function is 

$$ σ(x) = \frac{1}{1 + e^{-x}}. $$

And this is the graph for the sigmoid function:
![Sigmoid Function Graph](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/600px-Logistic-curve.svg.png)

Given this formula, how does a logistic regression model "learn"? The model learns by using an optimization algorithm, which is `lbfgs` for this model (and the default), to iteratively adjust its weights and bias to minimize a "Log-Loss" cost function, effectively finding the best-fitting sigmoid curve that separates the data classes.

Lastly, as there were more `ham`'s than `spam`'s, it creates an imbalance in the dataset, and in training it may introduce a bias for the most common label. Setting `class_weight="balanced"` makes sure that this bias is reduced.

## Pros
* Easy to interpret and efficient to train.
* Fast execution.
* Good accuracy.

## Cons
* Needs to be retrained for new spamming techniques.
* Sensitive to feature engineering as it does not understand context.
* Assumes linearity when making decisions.

# Use case
Let's say you run a platform that receives thousands of messages a day, whether that be emails, contact form submissions, or comments. Manually reviewing each message to determine whether it is spam is not feasible at scale.

You can use this API to automatically classify each message as spam or safe, allowing you to filter out unwanted content before it reaches your users or inbox.

Here is a list of text that this API could classify:
* Emails
* SMS messages
* Social media comments
* Forum posts
* Online reviews
* etc...