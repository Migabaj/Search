from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from tqdm import tqdm
import numpy as np
import torch
import json
import os


def newint(string):
    if string:
        return int(string)
    return 0


def cls_pooling(model_output):
    return model_output[0][:, 0]


def topn(query, corpus, tokenizer, model, cls_corpus, n=5):
    encoded_querys = tokenizer([query], padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_query_output = model(**encoded_querys)
    cls_query = cls_pooling(model_query_output)
    cos_sims = np.squeeze(cosine_similarity(cls_query, cls_corpus))
    return np.array(corpus)[np.argsort(cos_sims, axis=0)[:-(n+1):-1].ravel()]


def main():
    with open('../hw3/data/questions_about_love.jsonl', 'r') as f:
        corpus = list(f)

    answer_texts = [
        max(json.loads(corpus[i])["answers"],
            key=lambda x: newint(x["author_rating"]["value"]))["text"] for i in range(50000) if
        json.loads(corpus[i])["answers"]
    ]

    print("ЗАГРУЗКА ТОКЕНАЙЗЕРА...")
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    print("ЗАГРУЗКА МОДЕЛИ...")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

    if not os.path.exists("data/cls_answers.pt"):
        print("BERT-ЭМБЕДДИНГИ ОТВЕТОВ НЕ НАЙДЕНЫ. ЭМБЕДДИНГ...")
        encoded_answers = tokenizer(answer_texts[:50],
                                    padding=True,
                                    truncation=True,
                                    max_length=24,
                                    return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_answers)
        cls_answers = cls_pooling(model_output)
        torch.save(cls_answers, "data/cls_answers.pt")
        for i in tqdm(range(50, 300, 50)):
            cls_answers = torch.load("data/cls_answers.pt")
            encoded_answers_batch = tokenizer(answer_texts[i:i + 50],
                                              padding=True,
                                              truncation=True,
                                              max_length=24,
                                              return_tensors='pt')
            with torch.no_grad():
                model_output_batch = model(**encoded_answers_batch)
            cls_answers = torch.cat((cls_answers, cls_pooling(model_output_batch)), 0)
            torch.save(cls_answers, "data/cls_answers.pt")
    cls_answers = torch.load("data/cls_answers.pt")

    query_text = input("\nЗадайте Берту вопрос о любви: ")
    while True:
        if not query_text:
            break
        print("Берт думает...")
        print(*topn(query_text, answer_texts, tokenizer, model, cls_answers), sep="\n")
        query_text = input("\nЗадайте Берту вопрос о любви: ")


if __name__ == '__main__':
    main()
