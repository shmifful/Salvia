from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize

# Example text
text = """
Bangladesh, officially the People's Republic of Bangladesh, is a country in South Asia. It is the eighth-most populous country in the world and among the most densely populated with a population of over 174 million within an area of 148,460 square kilometres (57,320 sq mi). Bangladesh shares land borders with India to the north, west, and east, and Myanmar to the southeast. It has a coastline along the Bay of Bengal to its south and is separated from Bhutan and Nepal by the Siliguri Corridor, and from China by the Indian state of Sikkim to its north. Dhaka, the capital and largest city, is the nation's political, financial, and cultural centre. Chittagong is the second-largest city and the busiest port of the country.

The territory of modern Bangladesh was a stronghold of many Hindu and Buddhist dynasties in ancient history. Following the Muslim conquest in 1204, the region saw Sultanate and Mughal rule. As the largest subdivision of the Mughal Empire, the region of Bengal emerged one of the most prosperous and commercially active areas of the world, known for its thriving textile industry and agricultural productivity. The Battle of Plassey in 1757 marked the beginning of British colonial rule for the following two centuries. In the aftermath of the Partition of India in 1947, East Bengal became the eastern and most populous wing of the newly formed Dominion of Pakistan and was later renamed to East Pakistan.

Following over two decades of political repression and systemic racism from the West Pakistan–based government, East Pakistan experienced civil unrest in 1971, ultimately leading to a war for independence following a violent government military operation. The Mukti Bahini, with aid and assistance from Indian forces, waged a successful armed revolution; and despite a genocide perpetrated by Pakistan, Bangladesh became a sovereign nation on 16 December 1971. Post-Independence, Sheikh Mujibur Rahman led the country until his assassination in 1975. Presidency was later transferred to Ziaur Rahman, who himself was assassinated in 1981. The 1980s were dominated by the dictatorship of Hussain Muhammad Ershad, who was overthrown in a mass uprising in 1990. Following the democratisation in 1991, the Battle of the Begums between Khaleda Zia and Sheikh Hasina defined the country's politics for the next four decades. Hasina was overthrown in a mass uprising in August 2024.

Bangladesh is a unitary parliamentary republic based on the Westminster system. It is a middle power with the second-largest economy in South Asia. Bangladesh is home to the fourth-largest Muslim population in the world. It maintains the third-largest military in South Asia and is the largest contributor to the peacekeeping operations of the United Nations. Bangladesh consists of eight divisions, 64 districts, 495 sub-districts, and 4,578 union councils, and is home to the largest mangrove forest in the world. However, it has one of the largest refugee populations in the world and continues to face challenges such as endemic corruption, human rights abuses, political instability, and adverse effects of climate change. Bangladesh is a member state of SAARC and several other international organisations.
"""

def textrank(text, n=2):
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
    return " ".join([sentences[i] for i in top_indices])


shortened = textrank(text, 4)
print(shortened)