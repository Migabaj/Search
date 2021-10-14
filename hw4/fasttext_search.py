import os
import json
import gensim
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

curr_dir = os.getcwd()


def newint(string):
    if string:
        return int(string)
    return 0


def indexate(texts, model):
    print("ТОКЕНИЗАЦИЯ...")
    texts_tokenized = [gensim.utils.tokenize(text) for text in texts]

    print("ЭМБЕДДИНГ...")
    text_embs = []
    for toks in tqdm(texts_tokenized):
        try:
            text_embs.append(np.mean(model[toks], axis=0))
        except:
            text_embs.append(np.zeros(300))
    return np.array(text_embs)


def topn(query, corpus, corpus_emb, model, n=5):
    query_tokenized = gensim.utils.tokenize(query)
    query_emb = np.expand_dims(np.mean(model[query_tokenized], axis=0), axis=0)
    cos_sims = np.squeeze(cosine_similarity(query_emb, corpus_emb))
    return np.array(corpus)[np.argsort(cos_sims, axis=0)[:-(n+1):-1].ravel()][:]


def main():
    print("ЗАГРУЗКА МОДЕЛИ...")
    model = gensim.models.KeyedVectors.load('fasttext_model/araneum_none_fasttextcbow_300_5_2018.model')

    with open('../hw3/data/questions_about_love.jsonl', 'r') as f:
        corpus_str = "[" + ",".join(f.readlines()) + "]"
        corpus = json.loads(corpus_str)

    print("ПОДГОТОВКА ОТВЕТОВ...")
    answer_texts = [
        max(corpus[i]["answers"],
            key=lambda x: newint(x["author_rating"]["value"]))["text"] for i in range(50000) if corpus[i]["answers"]
    ]

    if not os.path.exists(os.path.join(curr_dir, "data/fasttext_answers.npy")):
        print("НЕ НАЙДЕНЫ ЭМБЕДДИНГИ ОТВЕТОВ. ЭМБЕДДИНГ...")
        answer_embs = indexate(answer_texts, model)

        print("СОХРАНЕНИЕ ЭМБЕДДИНГОВ...")
        with open(os.path.join(curr_dir, "data/fasttext_answers.npy"), "wb") as fw:
            np.save(fw, answer_embs)

    print("ЗАГРУЗКА ЭМБЕДДИНГОВ ОТВЕТОВ...")
    with open(os.path.join(curr_dir, "data/fasttext_answers.npy"), "rb") as f:
        answer_embs = np.load(f)

    query_text = input("\nЗадайте Фасттексту вопрос о любви: ")
    while True:
        if not query_text:
            break
        print("Фасттекст думает...")
        print(*topn(query_text, answer_texts, answer_embs, model), sep="\n")
        query_text = input("\nЗадайте Фасттексту вопрос о любви: ")


if __name__ == '__main__':
    main()
