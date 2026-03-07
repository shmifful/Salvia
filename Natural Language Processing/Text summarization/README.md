<details>
    <summary><b><u>Table of Contents (Click to expand)</u></b></summary>

  - [How to use](#how-to-use)
  - [Model structure](#model-structure)
  - [Use cases](#use-cases)
</details>

# How to use
## TextRank endpoint
```
https://api.salvia.dev/textrank/
```

## Input
```python
text: str
```

```python
n: int = 2
```

## Output
```python
dict : {
    "summary": str
}
```

## Raises 
`400`: No input text provided. 

## Usage
You can try this endpoint yourself (and many more) at [api.salvia.dev/docs](api.salvia.dev/docs).
### `python`
```py
import requests

URL = "https://api.salvia.dev/textrank/" 
text = """
Following over two decades of political repression and systemic racism from the West Pakistan–based government, East Pakistan experienced civil unrest in 1971, ultimately leading to a war for independence following a violent government military operation. The Mukti Bahini, with aid and assistance from Indian forces, waged a successful armed revolution; and despite a genocide perpetrated by Pakistan, Bangladesh became a sovereign nation on 16 December 1971. Post-Independence, Sheikh Mujibur Rahman led the country until his assassination in 1975. Presidency was later transferred to Ziaur Rahman, who himself was assassinated in 1981. The 1980s were dominated by the dictatorship of Hussain Muhammad Ershad, who was overthrown in a mass uprising in 1990. Following the democratisation in 1991, the Battle of the Begums between Khaleda Zia and Sheikh Hasina defined the country's politics for the next four decades. Hasina was overthrown in a mass uprising in August 2024.
""" 
res = requests.post(URL, , json={"text": text, "n": 2})

print(res.json())
# Output: {'summary': "Following the democratisation in 1991, the Battle of the Begums between Khaleda Zia and Sheikh Hasina defined the country's politics for the next four decades. Hasina was overthrown in a mass uprising in August 2024."}
```
`NOTE`: the API will return an error if the request is not a valid JSON.

# Model Structure
This is an unsupervised machine learning algorithm that follows the theory proposed in the ["TextRank: Bringing Order into Texts" by Rada Mihalcea and Paul Tarau](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) research paper. The algorithm implements a high-level abstraction of this theory. For a deeper understanding of the subject, please refer to the research paper.

`NOTE`: the example text used is the first 4 paragraphs of Wikipedia's page about [Bangladesh](https://en.wikipedia.org/wiki/Bangladesh).

This algorithm follows a few simple steps:

## 1. Tokenization of sentences
Since the algorithm identifies and returns the most important sentences in a given text, the first pre-processing step is sentence tokenization, which is splitting the text into a list of individual sentences.

```python
sentences = sent_tokenize(text)
```

## 2. Vectorization
Next, the machine is not able to understand raw texts, so each sentence must be vectorized, meaning they are converted from string to a numerical data such that it can be processed by the machine. In this process, the sentences are tokenized further at the word level, to make it easier to process. 

To further increase the efficiency of the algorithm, we can remove some words in the English language that hold very little contextual value (e.g., "the", "and", "is", etc...) by setting `stop_words="english"`.

```python
vectorizer = TfidfVectorizer(stop_words="english")
vec = vectorizer.fit_transform(sentences)
```

## 3. Calculating similarity between vectors
Then, we use the generated vectors to compute the similarity between each sentence. This will give us an `m x m` matrix of how similar each sentence is to another, where `m` is the number of sentences in the given text.

```python
similarity_matrix = cosine_similarity(vec)
```

## 4. Undirected Graph
From the similarity matrix, we can construct an undirected graph, where each `node` represents a sentence, and each `edge` represents the similarity score between two nodes.

```python
nx_graph = nx.from_numpy_array(similarity_matrix)
```

## 5. Ranking the importance of each sentence
Now that we have a graph, we can rank the importance of each sentence using the `PageRank` algorithm. A sentence is considered more important if it has high similarity to many other highly-ranked sentences, with the final score of each sentence being a reflection of both the number and weight of its connections.

```python
scores = nx.pagerank(nx_graph)
```

## Return
This algorithm returns the `n` most important sentences in the text in the order which they appear in the text. `n` is the number of most important sentences that the user requested for the text to be shortened to.

```python
top_indices = sorted(scores, key=scores.get, reverse=True)[:n] if n <= len(sentences) else sorted(scores, key=scores.get, reverse=True)

top_indices.sort()

return " ".join([sentences[i] for i in top_indices])
```

## Pros
* Lightweight.
* Does not require any labelled data.
* Could work for any language technically (improve the efficiency for all languages with minor tweaks).
* Summary is actual reflection of the original text.
* Domain independent and can work with any type of text (e.g., news, email, research papers).

## Cons
* Sentences can feel disjointed and lack flow when reading.
* Longer sentences tend to have higher similarity scores.
* Relies on statistics/does not understand context.
* Cannot merge sentences for a more cohesive structure.

# Use cases 
Let's say you are doing research for a machine learning project, and you have 20 articles and research papers with complex terminology to read. Just skim reading each paper does not work as you accidentally miss the important parts of the texts. How do you cut through the complexity and get to the core ideas of all these bodies of text?

You can use this API to extract the most important ideas from the text, giving an overview of the text as a whole, so that it is easier to understand. And if you require further clarification, then you can always increase the number of sentences you want to read from these texts, or read the whole text.

Here is a list of text that this API could summarize:
* News articles
* Research papers
* Emails/newsletters
* Wikipedia pages
* (Long DMs from your friends👀)
* etc...