import os
import json
import gensim
import torch
import numpy as np
from scipy import sparse
from pymystem3 import Mystem
from string import punctuation
from tqdm import tqdm
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

punctuation += " “‘”’«»1234567890"
curr_dir = os.getcwd()


def newint(string):
    if string:
        return int(string)
    return 0


def preprocess(corpus):
    clean_corpus = []
    ana = Mystem()
    print("ПРОИСХОДИТ ПРЕДОБРАБОТКА КОРПУСА...")
    for text in tqdm(corpus):
        tokens = ana.lemmatize(text)
        tokens = [tok for tok in tokens
                  if not any(elem in tok for elem in punctuation) and
                  tok not in stopwords.words("russian")]
        clean_corpus.append(" ".join(tokens).lower())
    print()
    return clean_corpus


def cls_pooling(corpus, mdl, tknzr):
    t = tknzr(corpus, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = mdl(**t)
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings.cpu().numpy()


def indexate(corpus, vctrzr):
    try:
        check_is_fitted(vctrzr)
        return vctrzr.transform(corpus)
    except:
        return vctrzr.fit_transform(corpus)


def bm25_indexate(corpus, count_vectorizer, k=2, b=0.75):
    tf_idf_vctrzr = TfidfVectorizer(use_idf=True)
    tf_vctrzr = TfidfVectorizer(use_idf=False)

    count_matrix = count_vectorizer.fit_transform(corpus)
    if os.path.exists(os.path.join(curr_dir, '../hw3/data/sparse.npz')):
        return sparse.load_npz(os.path.join(curr_dir, '../hw3/data/sparse.npz'))
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


def fasttext_indexate(texts, model):
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


def bm25(query, corpus, count_vectorizer):
    corpus_matrix = bm25_indexate(corpus, count_vectorizer)
    query_matrix = count_vectorizer.transform(query)
    return np.dot(query_matrix, corpus_matrix.T).toarray()


def score(evaluation_matrix, n=5):
    matrix_argsort_n = np.argsort(evaluation_matrix, axis=1)[:, :-(n+1):-1]
    question_range = np.expand_dims(np.arange(evaluation_matrix.shape[0]), axis=1)
    question_in_topn = np.sum(question_range == matrix_argsort_n, axis=1)
    return np.sum(question_in_topn) / evaluation_matrix.shape[0]


def main():
    print("ЗАГРУЗКА КОРПУСА...")
    with open(os.path.join(curr_dir,
                           '../hw3/data/questions_about_love.jsonl'),
              'r') as f:
        corpus_str = "[" + ",".join(f.readlines()) + "]"
        corpus = json.loads(corpus_str)

    answer_texts = [
        max(corpus[i]["answers"],
            key=lambda x: newint(x["author_rating"]["value"]))["text"] for i in range(10000) if corpus[i]["answers"]
    ]

    question_texts = [question["question"] + " " + question["comment"] for question in corpus[:10000] if
                      question["answers"]]

    if os.path.exists(os.path.join(curr_dir,
                                   "data/answers_preproc.json")):
        print("ЗАГРУЗКА ПРЕДОБРАБОТАННЫХ ОТВЕТОВ...")
        with open(os.path.join(curr_dir, "data/answers_preproc.json"),
                  "r",
                  encoding="utf-8") as f:
            answers_preproc = json.load(f)
    else:
        print("ПРЕДОБРАБОТАННЫЕ ОТВЕТЫ НЕ НАЙДЕНЫ.")
        answers_preproc = preprocess(answer_texts)

    if os.path.exists(os.path.join(curr_dir,
                                   "data/questions_preproc.json")):
        print("ЗАГРУЗКА ПРЕДОБРАБОТАННЫХ ВОПРОСОВ...")
        with open(os.path.join(curr_dir, "data/questions_preproc.json"),
                  "r",
                  encoding="utf-8") as f:
            questions_preproc = json.load(f)
    else:
        print("ПРЕДОБРАБОТАННЫЕ ВОПРОСЫ НЕ НАЙДЕНЫ.")
        questions_preproc = preprocess(question_texts)

    # ИНДЕКСАЦИЯ
    print("\nИНДЕКСАЦИЯ...")
    ## CountVectorizer
    print("CountVectorizer...")
    cnt_vctrzr = CountVectorizer()
    cnt_answer_embs = indexate(answers_preproc, cnt_vctrzr)
    cnt_question_embs = indexate(questions_preproc, cnt_vctrzr)

    ## TfidfVectorizer
    print("TfidfVectorizer...")
    tfidf_vctrzr = TfidfVectorizer()
    tfidf_answer_embs = indexate(answers_preproc, tfidf_vctrzr)
    tfidf_question_embs = indexate(questions_preproc, tfidf_vctrzr)

    ## FastText
    print("FastText...")
    model = None
    if not os.path.exists(os.path.join(curr_dir, "data/fasttext_answers.npy")):
        print("НЕ НАЙДЕНЫ ЭМБЕДДИНГИ ОТВЕТОВ. ЭМБЕДДИНГ...")
        try:
            model = gensim.models.KeyedVectors.load('fasttext_model/araneum_none_fasttextcbow_300_5_2018.model')
        except FileNotFoundError:
            raise FileNotFoundError('Должна быть скачана fasttext-модель araneum_none_fasttextcbow_300_5_2018 и добавлена в директорию fasttext_model')
        fasttext_answer_embs = fasttext_indexate(answer_texts, model)

        print("СОХРАНЕНИЕ ЭМБЕДДИНГОВ...")
        with open(os.path.join(curr_dir, "data/fasttext_answers.npy"), "wb") as fw:
            np.save(fw, fasttext_answer_embs)

    print("ЗАГРУЗКА ЭМБЕДДИНГОВ ОТВЕТОВ...")
    with open(os.path.join(curr_dir, "data/fasttext_answers.npy"), "rb") as f:
        fasttext_answer_embs = np.load(f)

    if not os.path.exists(os.path.join(curr_dir, "data/fasttext_questions.npy")):
        print("НЕ НАЙДЕНЫ ЭМБЕДДИНГИ ВОПРОСОВ. ЭМБЕДДИНГ...")
        if not model:
            model = gensim.models.KeyedVectors.load('fasttext_model/araneum_none_fasttextcbow_300_5_2018.model')
        fasttext_question_embs = fasttext_indexate(question_texts, model)

        print("СОХРАНЕНИЕ ЭМБЕДДИНГОВ...")
        with open(os.path.join(curr_dir, "data/fasttext_questions.npy"), "wb") as fw:
            np.save(fw, fasttext_question_embs)

    print("ЗАГРУЗКА ЭМБЕДДИНГОВ ВОПРОСОВ...")
    with open(os.path.join(curr_dir, "data/fasttext_questions.npy"), "rb") as f:
        fasttext_question_embs = np.load(f)


    ## BERT
    print("BERT...")
    if not os.path.exists("data/cls_answers.pt") or not os.path.exists("data/cls_questions.pt"):
        tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
        model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

    if os.path.exists("data/cls_answers.pt"):
        bert_answer_embs = torch.load("data/cls_answers.pt")
    else:
        print("BERT-ЭМБЕДДИНГИ ОТВЕТОВ НЕ НАЙДЕНЫ. ЭМБЕДДИНГ...")
        encoded_answers = tokenizer(answer_texts[:50],
                                    padding=True,
                                    truncation=True,
                                    max_length=24,
                                    return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_answers)
        cls = cls_pooling(model_output)
        torch.save(cls, "data/cls_answers.pt")
        for i in tqdm(range(50, 10000, 50)):
            cls = torch.load("data/cls_answers.pt")
            encoded_answers_batch = tokenizer(answer_texts[i:i+50],
                                              padding=True,
                                              truncation=True,
                                              max_length=24,
                                              return_tensors='pt')
            with torch.no_grad():
                model_output_batch = model(**encoded_answers_batch)
            cls = torch.cat((cls, cls_pooling(model_output_batch)), 0)
            torch.save(cls, "data/cls_answers.pt")
            del cls

    if os.path.exists("data/cls_questions.pt"):
        bert_question_embs = torch.load("data/cls_questions.pt")
    else:
        print("BERT-ЭМБЕДДИНГИ ВОПРОСОВ НЕ НАЙДЕНЫ. ЭМБЕДДИНГ...")
        encoded_questions = tokenizer(question_texts[:50],
                                      padding=True,
                                      truncation=True,
                                      max_length=24,
                                      return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_questions)
        cls = cls_pooling(model_output)
        torch.save(cls, "data/cls_questions.pt")
        for i in tqdm(range(50, 10000, 50)):
            cls = torch.load("data/cls_questions.pt")
            encoded_questions_batch = tokenizer(question_texts[i:i+50],
                                                padding=True,
                                                truncation=True,
                                                max_length=24,
                                                return_tensors='pt')
            with torch.no_grad():
                model_output_batch = model(**encoded_questions_batch)
            cls = torch.cat((cls, cls_pooling(model_output_batch)), 0)
            torch.save(cls, "data/cls_questions.pt")
            del cls

    # ПОДСЧЕТ КАЧЕСТВА
    print("\nПОДСЧЕТ КАЧЕСТВА...")

    ## CountVectorizer
    print("CountVectorizer...")
    cnt_answer_embs_batch = cnt_answer_embs[:10000]
    cnt_question_embs_batch = cnt_question_embs[:10000]
    cnt_cos_sim = cosine_similarity(cnt_question_embs_batch, cnt_answer_embs_batch)
    cnt_score_5 = score(cnt_cos_sim)
    cnt_score_1 = score(cnt_cos_sim, 1)

    ## TfidfVectorizer
    print("TfidfVectorizer...")
    tfidf_answer_embs_batch = tfidf_answer_embs[:10000]
    tfidf_question_embs_batch = tfidf_question_embs[:10000]
    tfidf_cos_sim = cosine_similarity(tfidf_question_embs_batch, tfidf_answer_embs_batch)
    tfidf_score_5 = score(tfidf_cos_sim)
    tfidf_score_1 = score(tfidf_cos_sim, 1)

    ## BM25
    print("BM25...")
    bm_cnt_vctrzr = CountVectorizer()
    bm25_matrix = bm25(questions_preproc[:10000], answers_preproc[:10000], bm_cnt_vctrzr)
    bm25_score_5 = score(bm25_matrix)
    bm25_score_1 = score(bm25_matrix, 1)

    ## FastText
    print("FastText...")
    fasttext_answer_embs_batch = fasttext_answer_embs[:10000]
    fasttext_question_embs_batch = fasttext_question_embs[:10000]
    fasttext_cos_sim = cosine_similarity(fasttext_question_embs_batch, fasttext_answer_embs_batch)
    fasttext_score_5 = score(fasttext_cos_sim)
    fasttext_score_1 = score(fasttext_cos_sim, 1)

    ## BERT
    print("BERT...")
    bert_question_embs_batch = bert_question_embs[:10000]
    bert_answer_embs_batch = bert_answer_embs[:10000]
    bert_cos_sim = cosine_similarity(bert_question_embs_batch, bert_answer_embs_batch)
    bert_score_5 = score(bert_cos_sim)
    bert_score_1 = score(bert_cos_sim, 1)

    print("\nРЕЗУЛЬТАТЫ:")
    print("method\t(n=5)\t(n=1)")
    print(f"CountVectorizer\t{cnt_score_5}\t{cnt_score_1}")
    print(f"TfidfVectorizer\t{tfidf_score_5}\t{tfidf_score_1}")
    print(f"BM25\t{bm25_score_5}\t{bm25_score_1}")
    print(f"FastText\t{fasttext_score_5}\t{fasttext_score_1}")
    print(f"BERT\t{bert_score_5}\t{bert_score_1}")


if __name__ == '__main__':
    main()
