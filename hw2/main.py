import os
from tqdm import tqdm
from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

punctuation += " “‘”’«»1234567890"
vectorizer = TfidfVectorizer(analyzer='word')
ana = Mystem()

curr_dir = os.getcwd()
# наш корпус.
friends_corpus = []
# упорядоченный список названий серий.
friends_episode_names = []

titles_string = "\n9999\n00:00:0,500 --> 00:00:2,00\nwww.tvsubtitles.net\n"
for root, dirs, files in os.walk(os.path.join(curr_dir, '../hw1/friends-data')):
    for name in files:
        if not name.startswith("."):
            friends_episode_names.append(name)
            with open(os.path.join(root, name), 'r') as f:
                friends_corpus.append(f.read().strip(titles_string))


# все та же функция для препроцессинга.
# из-за того, что это майстем, она довольно
# медленная.
def preprocess(corpus):
    clean_corpus = []
    print("ПРОИСХОДИТ ПРЕДОБРАБОТКА КОРПУСА:")
    for text in tqdm(corpus):
        tokens = ana.lemmatize(text)
        tokens = [tok for tok in tokens
                  if not any(elem in tok for elem in punctuation) and
                  tok not in stopwords.words("russian")]
        clean_corpus.append(" ".join(tokens).lower())
    print()
    return clean_corpus


friends_corpus = preprocess(friends_corpus)


# берет: корпус.
# выдает: матрицу term-document (TF-IDF).
def indexate(corpus):
    return vectorizer.fit_transform(corpus).toarray()


# создаем матрицу.
X = indexate(friends_corpus)


# берет: текст.
# выдает: матрицу term-document
# с одним документом -- текстом-аргументом.
def indexate_line(line):
    return vectorizer.transform([line]).toarray()


# берет: текст и матрицу term-document.
# выдает: список косинусных расстояний запроса
# от каждого документа.
def compare(text, term_document_matrix):
    searchvec = indexate_line(text)
    return cosine_similarity(term_document_matrix, searchvec)


# берет: запрос.
# выдает: упорядоченный список названий серий,
# которые максимально подходят к
# лемматизированному запросу.
def search(text):
    text = " ".join(ana.lemmatize(text))
    comparisons = compare(text, X)
    episodes = [x for _, x in sorted(zip(comparisons, friends_episode_names))[::-1]]
    return episodes


# демонстрирует работу кода.
def main():
    line = "a"
    while line != "":
        line = input("Введите запрос: ")
        print("САМЫЕ ПОДХОДЯЩИЕ СЕРИИ:")
        for i, ep in enumerate(search(line)[:10]):
            print(str(i+1)+".", ep)
        print()


if __name__ == '__main__':
    main()

# made by nejenek
