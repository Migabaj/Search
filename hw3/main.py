import os
import json
import numpy as np
from scipy import sparse
from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


punctuation += " “‘”’«»1234567890"
count_vctrzr = CountVectorizer()
curr_dir = os.getcwd()


def preprocess(corpus):
    clean_corpus = []
    ana = Mystem()
    for text in corpus:
        tokens = ana.lemmatize(text)
        tokens = [tok for tok in tokens
                  if not any(elem in tok for elem in punctuation) and
                  tok not in stopwords.words("russian")]
        clean_corpus.append(" ".join(tokens).lower())
    print()
    return clean_corpus


def bm25_indexate(corpus, k=2, b=0.75):

    tf_idf_vctrzr = TfidfVectorizer(use_idf=True)
    tf_vctrzr = TfidfVectorizer(use_idf=False)

    count_matrix = count_vctrzr.fit_transform(corpus)
    if os.path.exists(os.path.join(curr_dir, 'data/sparse.npz')):
        return sparse.load_npz(os.path.join(curr_dir, 'data/sparse.npz'))
    tfidf_matrix = tf_idf_vctrzr.fit_transform(corpus)
    tf = tf_vctrzr.fit_transform(corpus)

    len_d = count_matrix.sum(axis=1)
    avdl = len_d.mean()
    idf = np.expand_dims(tf_idf_vctrzr.idf_, axis=0)
    denom_component = np.expand_dims(k * (1 - b + b * len_d / avdl), axis=-1)

    values = []
    rows = []
    cols = []

    for i, j in zip(*tf.nonzero()):
        numerator = tf[i, j] * idf[0][j] * (k + 1)
        denominator = tf[i, j] + denom_component[i]
        value = numerator / denominator
        values.append(value[0][0])
        rows.append(i)
        cols.append(j)

    return sparse.csr_matrix((values, (rows, cols)))


def bm25(query, corpus):
    corpus_matrix = bm25_indexate(corpus)
    query = preprocess([query])[0]
    query_vector = count_vctrzr.transform([query])

    return np.dot(corpus_matrix, query_vector.T)


def author_value(item):
    if item["author_rating"]["value"]:
        return int(item["author_rating"]["value"])
    return 0


def closest_docs(query):
    with open(os.path.join(curr_dir, 'data/questions_about_love.jsonl')) as f:
        questions_data = list(f)
    answers_corpus = [
        max(json.loads(questions_data[i])["answers"],
            key=lambda x: author_value(x))["text"] for i in range(50000) if json.loads(questions_data[i])["answers"]
    ]
    if os.path.exists(os.path.join(curr_dir, 'data/text.json')):
        with open(os.path.join(curr_dir, 'data/text.json'),
                  "r",
                  encoding="utf-8") as f:
            corpus = np.array(json.load(f))
    else:
        print("ФАЙЛ С КОРПУСОМ ОТСУТСТВУЕТ. ПРЕДОБРАБОТКА...")
        corpus = preprocess(answers_corpus)
        with open(os.path.join(curr_dir, 'data/text.json'),
                  "w",
                  encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False)
        corpus = np.array(corpus)

    bm25_ind = bm25(query, corpus)
    return np.array(answers_corpus)[np.argsort(bm25_ind.toarray(), axis=0)[::-1].ravel()]


# ДЕМОНСТРАЦИЯ
if __name__ == "__main__":
    print(closest_docs("мама")[:5])

