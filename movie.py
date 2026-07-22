import nltk
import string
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from gensim import corpora, models, similarities
from gensim.models import Word2Vec

nltk.download("punkt")
nltk.download("punkt_tab")   
nltk.download("stopwords")

reviews = [
    "This movie was amazing and I loved every minute of it.",
    "The acting was terrible and the story was boring.",
    "It was an average movie with some good scenes.",
    "Fantastic visuals and great performances by the cast.",
    "I hated this film because it was too long.",
    "The movie was okay, nothing special but not bad.",
    "Excellent plot with brilliant acting and music.",
    "Too much psychology and not enough pop. It is possible to be too serious, you know"
]

def preprocess(reviews):
    stop_words = set(stopwords.words("english"))

    tokens = []
    all_words = []

    for review in reviews:
        words = word_tokenize(review.lower())
        words = [w for w in words if w not in string.punctuation]
        words = [w for w in words if w not in stop_words]
        tokens.append(words)
        all_words.extend(words)

    counts = Counter(all_words)

    cleaned = []
    for review in tokens:
        cleaned.append([w for w in review if counts[w] > 1])

    return cleaned

clean_reviews = preprocess(reviews)

dictionary = corpora.Dictionary(clean_reviews)
corpus = [dictionary.doc2bow(text) for text in clean_reviews]

print("\n----- Sentiment Analysis -----")

positive = negative = neutral = 0

for review in reviews:
    polarity = TextBlob(review).sentiment.polarity

    if polarity > 0:
        sentiment = "Positive"
        positive += 1
    elif polarity < 0:
        sentiment = "Negative"
        negative += 1
    else:
        sentiment = "Neutral"
        neutral += 1

    print("\nReview:", review)
    print("Sentiment:", sentiment)
    print("Polarity:", round(polarity, 2))

plt.bar(
    ["Positive", "Negative", "Neutral"],
    [positive, negative, neutral],
    color=["green", "red", "gray"]
)
plt.title("Movie Review Sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.show()

print("\n----- LDA Topic Modeling -----")

lda = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=2,
    passes=10,
    random_state=1
)

for i in range(2):
    print(f"\nTopic {i+1}:")
    print(lda.print_topic(i))

print("\nTopic Scores for Each Review")
for i, doc in enumerate(corpus):
    print(f"Review {i+1}: {lda.get_document_topics(doc)}")

print("\n----- LSI Review Similarity -----")

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
index = similarities.MatrixSimilarity(lsi[corpus])

for i, doc in enumerate(corpus):
    sims = list(enumerate(index[lsi[doc]]))
    sims[i] = (i, -1)  # Ignore itself
    best = max(sims, key=lambda x: x[1])

    print(f"Review {i+1} is most similar to Review {best[0]+1}")
    print("Similarity Score:", round(best[1], 3))

print("\n----- Word2Vec -----")

model = Word2Vec(
    sentences=clean_reviews,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1
)

while True:
    word = input("\nEnter a word (or 'quit'): ").lower()

    if word == "quit":
        break

    if word in model.wv:
        print(model.wv.most_similar(word))
    else:
        print("Word not found in vocabulary.")