# coding=utf-8
import sys,os
import pandas as pd
sys.path.append('.')
from transformers import BertTokenizer, BertModel, AlbertModel, BertForSequenceClassification, \
    AlbertForSequenceClassification
from model import CDNForCLSModel
import torch
# tokenizer_class, model_class = (BertTokenizer, BertModel)
from utils import seed_everything, ProgressBar, TokenRematch
import json
import numpy as np
from tqdm import tqdm
import jieba
from gensim import corpora, models, similarities
from data import CDNDataset
from torch.utils.data import Dataset, DataLoader


def _get_labels():
    df = pd.read_excel("./RAW_DATA/国际疾病分类 ICD-10北京临床版v601.xlsx", header=None)
    normalized_word = df[1].unique().tolist()
    label2id = {word: idx for idx, word in enumerate(normalized_word)}
    id2label = {idx: word for idx, word in enumerate(normalized_word)}
    return label2id, id2label
def _init_label_embedding(label2id,id2label):
    all_label_list = []
    for idx in range(len(label2id.keys())):
        all_label_list.append(list(jieba.cut(id2label[idx])))

    dictionary = corpora.Dictionary(all_label_list)  # 词典
    corpus = [dictionary.doc2bow(doc) for doc in all_label_list]  # 语料库
    tfidf = models.TfidfModel(corpus)  # 建立模型
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))

    return dictionary, index, tfidf
def _recall(texts,recall_k,dictionary, index, tfidf,label2id):
    recall_scores_idx = np.zeros((len(texts), recall_k), dtype=np.int)

    recall_scores = np.zeros((len(texts), recall_k))
    for i, x in tqdm(enumerate(texts), total=len(texts)):
        x_split = list(jieba.cut(x))
        x_vec = dictionary.doc2bow(x_split)
        x_sim = index[tfidf[x_vec]]  # 相似度分数 (1, labels)

        x_dices = np.zeros(len(label2id.keys()))
        x_set = set(x)

        for j, y in enumerate(label2id.keys()):
            y_set = set(y)
            x_dices[j] = len(x_set & y_set) / min(len(x_set), len(y_set))

        x_scores = x_sim + x_dices
        x_scores_idx = np.argsort(x_scores)[:len(x_scores) - recall_k - 1:-1]  # 由大到小排序,取前K个
        x_scores = np.sort(x_scores)[:len(x_scores) - recall_k - 1:-1]
        recall_scores[i] += x_scores
        recall_scores_idx[i] += x_scores_idx
    return recall_scores_idx, recall_scores


def _save_cache(task_data_dir,outputs, recall_orig_samples, mode='train'):
    cache_df = pd.DataFrame(outputs)
    cache_df.to_csv(os.path.join(task_data_dir, f'{mode}_samples.csv'), index=False)
    recall_orig_cache_df = pd.DataFrame(recall_orig_samples)
    recall_orig_cache_df['label'] = recall_orig_cache_df.label.apply(lambda x: " ".join([str(i) for i in x]))
    recall_orig_cache_df['recall_label'] = recall_orig_cache_df.recall_label.apply(
        lambda x: " ".join([str(i) for i in x]))
    recall_orig_cache_df.to_csv(os.path.join(task_data_dir, f'{mode}_recall_orig_samples.csv'),
                                index=False)

def _load_cache(task_data_dir,mode='train'):
        outputs = {'text1': [], 'text2': [], 'label': []}
        recall_orig_samples = {'text': [], 'label': [], 'recall_label': []}

        train_cache_df = pd.read_csv(os.path.join(task_data_dir, f'{mode}_samples.csv'))
        outputs['text1'] = train_cache_df['text1'].values.tolist()
        outputs['text2'] = train_cache_df['text2'].values.tolist()
        outputs['label'] = train_cache_df['label'].values.tolist()

        train_recall_orig_cache_df = pd.read_csv(os.path.join(task_data_dir, f'{mode}_recall_orig_samples.csv'))
        recall_orig_samples['text'] = train_recall_orig_cache_df['text'].values.tolist()
        recall_orig_samples['label'] = train_recall_orig_cache_df['label'].values.tolist()
        recall_orig_samples['recall_label'] = train_recall_orig_cache_df['recall_label'].values.tolist()
        recall_orig_samples['label'] = [[int(label) for label in str(recall_orig_samples['label'][i]).split()] for i in
                                        range(len(recall_orig_samples['label']))]
        recall_orig_samples['recall_label'] = [[int(label) for label in str(recall_orig_samples['recall_label'][i]).split()] for i in
                                               range(len(recall_orig_samples['recall_label']))]
        recall_samples_scores = np.load(os.path.join(self.task_data_dir, f'{mode}_recall_score.npy'))

        return outputs, recall_orig_samples, recall_samples_scores


def _get_cls_samples(task_data_dir,orig_samples, mode,id2label,recall_samples_idx, recall_samples_scores):
    outputs = {'text1': [], 'text2': [], 'label': []}

    texts = [orig_samples]
    # recall_samples_idx, recall_samples_scores = _recall(text,200,dictionary, index, tfidf,label2id)
    np.save(os.path.join(task_data_dir, f'{mode}_recall_score.npy'), recall_samples_scores)
    recall_orig_samples = {'text': [], 'label': [], 'recall_label': []}

    for text, recall_label in zip(texts, recall_samples_idx):

        recall_orig_samples['text'].append(text)
        recall_orig_samples['recall_label'].append(recall_label)
        recall_orig_samples['label'].append([0])

        for label_ in recall_label:
            outputs['text1'].append(text)
            outputs['text2'].append(id2label[label_])
            outputs['label'].append(0)
    _save_cache(task_data_dir,outputs, recall_orig_samples,'test')

    return outputs, recall_orig_samples, recall_samples_scores

def get_test_dataloader(test_dataset, batch_size=None):
    if not batch_size:
        batch_size = 30

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
def predict(test_dataset, model,tokenizer):
        test_dataset.text1 = test_dataset.text1
        test_dataset.text2 = test_dataset.text2
        test_dataloader = get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)

        preds = None
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Evaluation')
        for step, item in enumerate(test_dataloader):
            print(step)
            model.eval()

            text1 = item[0]
            text2 = item[1]


            inputs = tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                        truncation='longest_first', max_length=64)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs

            if preds is None:
                preds = logits.detach().softmax(-1)[:, 1].cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().softmax(-1)[:, 1].cpu().numpy(), axis=0)

            pbar(step, info="")

        preds = preds.reshape(len(preds) // 200, 200)
        np.save(os.path.join("./data_predict", f'cdn_test_preds.npy'), preds)
        return preds
def _get_num_samples(orig_sample):
    outputs = {'text1': [], 'text2': [], 'label': []}
    outputs['text1'] = orig_sample['text']

    return outputs
###加载cls模型

tokenizer = BertTokenizer.from_pretrained( './data_model/cls')
ngram_dict = None
with torch.no_grad():
    model = CDNForCLSModel(BertModel,encoder_path='./data_model/cls',num_labels=2)
    model.load_state_dict(torch.load("./data_model/pytorch_model_cls.pt"))
with torch.no_grad():
    model_n = BertForSequenceClassification.from_pretrained('./data_model/num',num_labels=3)
def cdn_commit_prediction(text, preds, num_preds, recall_labels, recall_scores, output_dir, id2label):
    text1 = text

    pred_result = []
    active_indices = (preds >= 0.5)
    for text, active_indice, pred, num, recall_label, recall_score in zip(text1, active_indices, preds, num_preds, recall_labels, recall_scores):
        tmp_dict = {'text': text, 'normalized_result': []}
        print(active_indice)
        final_pred = pred[active_indice]
        recall_score = recall_score[active_indice]
        recall_label = recall_label[active_indice]

        if len(final_pred):
            final_score = (recall_score / 2 + final_pred) / 2
            final_score = np.argsort(final_score)[::-1]
            recall_label = recall_label[final_score]

            num = num + 1
            ji, ban, dou = text.count("及"), text.count("伴"), text.count(";")
            if (ji + ban + dou + 1) > num:
                num = ji + ban + dou + 1
            if num == 1:
                tmp_dict['normalized_result'].append(recall_label[0])
            elif num == 2:
                tmp_dict['normalized_result'].extend(recall_label[:2].tolist())
            else:
                sum_ = max((ji + ban + dou + 1), num, 3)
                tmp_dict['normalized_result'].extend(recall_label[:sum_].tolist())
            tmp_dict['normalized_result'] = [id2label[idx] for idx in tmp_dict['normalized_result']]

        if len(tmp_dict['normalized_result']) == 0:
            tmp_dict['normalized_result'] = [text]
        tmp_dict['normalized_result'] = "##".join(tmp_dict['normalized_result'])
        pred_result.append(tmp_dict)

    with open(os.path.join(output_dir, 'test_.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))
    return pred_result
if __name__=="__main__":
    text = "髂腰肌囊性占位"
    task_data_dir="./data_predict"
    label2id, id2label=_get_labels()
    dictionary, index, tfidf= _init_label_embedding(label2id, id2label)
    recall_samples_idx, recall_samples_scores = _recall(text,200,dictionary, index, tfidf,label2id)
    # print(recall_samples_idx,recall_samples_scores)
    outputs, recall_orig_samples, recall_samples_scores=_get_cls_samples(task_data_dir,text, 'test',id2label,recall_samples_idx, recall_samples_scores)
    test_dataset = CDNDataset(outputs, "data_processor", dtype='cls', mode='test')
    cls_preds = predict(test_dataset,model,tokenizer)
    print(cls_preds.shape)
    ## 第二步预测有几个num；返回最像的
    inputs = tokenizer([text],padding='max_length', max_length=64,
                                        truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model_n(**inputs)
    logits = outputs[0]
    print(logits)
    preds = logits.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
    print(preds)
    finall = cdn_commit_prediction([text],cls_preds,preds,recall_orig_samples['recall_label'], recall_samples_scores,"./", id2label)
    print(finall)