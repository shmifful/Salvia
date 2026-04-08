# Salvia API
Salvia is an open-source machine learning project providing endpoints for various algorithms. Each endpoint is documented with an explanation of how the underlying model works, making it a useful resource for both practical use and learning.

Currently, these are the live endpoints:

* [Sentiment Analysis](https://github.com/shmifful/Salvia/tree/main/Natural%20Language%20Processing/Sentiment%20analysis) - analyzes a given text and returns whether the text is positive, neutral, or negative.
* [Spam Filtering](https://github.com/shmifful/Salvia/tree/main/Natural%20Language%20Processing/Spam%20classifier) - recognizes if a given text is spam or not.
* [Text Summarization](https://github.com/shmifful/Salvia/tree/main/Natural%20Language%20Processing/Text%20summarization) - extracts the most important `n` sentences from a given text.

You can also try these endpoints on the interactive [documentation](https://api.salvia.dev/docs/).

## Tech stack
`FastAPI`: used to build the endpoints for each model.

`scikit-learn`: used to build the models.

`NLTK`: used to pre-process strings.

`NetworkX`: used to build graphs.

`Docker`: used to containerize the API.

`Google Cloud`: used to host the API on the cloud.

`Kaggle`: used to collect data to train the models.