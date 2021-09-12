import os
import numpy as np
from tqdm import tqdm
from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

punctuation += " “‘”’«»1234567890"


# берет: корпус текстов.
# выдает: обработанный корпус текстов,
# без знаков препинания, чисел, стоп-слов и верхнего регистра.
# (конкретно майстем здесь работает медленно)
def preprocess(corpus):
    clean_corpus = []
    ana = Mystem()
    print("ПРОИСХОДИТ ПРЕДОБРАБОТКА КОРПУСА:")
    for text in tqdm(corpus):
        tokens = ana.lemmatize(text)
        tokens = [tok for tok in tokens
                  if not any(elem in tok for elem in punctuation) and
                  tok not in stopwords.words("russian")]
        clean_corpus.append(" ".join(tokens).lower())
    print()
    return clean_corpus


# берет: корпус и модель.
# выдает: матрицу term-document.
def indexate(corpus, vctrzr):
    return vctrzr.fit_transform(corpus)


def main():
    curr_dir = os.getcwd()
    friends_corpus = []

    # составление корпуса текстов в виде питоновского списка.
    titles_string = "\n9999\n00:00:0,500 --> 00:00:2,00\nwww.tvsubtitles.net\n"
    for root, dirs, files in os.walk(os.path.join(curr_dir, 'friends-data')):
        for name in files:
            if not name.startswith("."):
                with open(os.path.join(root, name), 'r') as f:
                    friends_corpus.append(f.read().strip(titles_string))

    # обрабатывает и индексирует корпус.
    friends_corpus = preprocess(friends_corpus)
    vectorizer = CountVectorizer(analyzer='word')
    X = indexate(friends_corpus, vectorizer)

    # задание 2a
    matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    matrix_freq_sorted = sorted(list(range(len(matrix_freq))),
                                key=lambda num: matrix_freq[num])[::-1]
    print("САМОЕ ЧАСТОТНОЕ СЛОВО КОРПУСА:")
    print(vectorizer.get_feature_names()[matrix_freq_sorted[0]] + "\n")

    # задание 2b
    print("САМОЕ РЕДКОЕ СЛОВО КОРПУСА:")
    print(vectorizer.get_feature_names()[matrix_freq_sorted[-1]] + "\n")

    # задание 2c
    print("СЛОВА, КОТОРЫЕ ЕСТЬ В КАЖДОЙ СЕРИИ:")
    for i, arr in enumerate(X.toarray().T):
        if np.all(arr):
            print(vectorizer.get_feature_names()[i])
    print()

    # задание 2d

    # берет: слово.
    # выдает: кол-во упоминаний слова во всех документах корпуса.
    def total_count(word):
        word_ix = vectorizer.vocabulary_.get(word)
        word_counts = X.toarray().T[word_ix]
        return np.sum(word_counts)

    print("КОЛИЧЕСТВО УПОМИНАНИЙ ДРУЗЕЙ НА ПРОТЯЖЕНИИ 7 СЕЗОНОВ:")
    print("Моника:", total_count("моника") +
          total_count("мон"))
    print("Рейчел:", total_count("рейчел") +
          total_count("рейч"))
    print("Фиби:", total_count("фиби") +
          total_count("фибс"))
    print("Росс:", total_count("росс"))
    print("Чендлер:", total_count("чендлер") +
          total_count("чэндлер") +
          total_count("чен"))
    print("Джоуи:", total_count("джоуи") +
          total_count("джо"))
    print("Победитель, соответственно, Росс.")


if __name__ == '__main__':
    main()

# made by nejenek