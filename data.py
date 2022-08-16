# coding=utf-8
import os,sys
import json
import numpy as np
from torch.utils.data import Dataset
import torch
import jieba
import pandas as pd
from gensim import corpora, models, similarities
from tqdm import tqdm
from utils import str_q2b,load_json
'''

数据格式如下：
  {
    "text": "子宫颈鳞癌IIB期放化疗后",
    "normalized_result": "子宫颈恶性肿瘤##鳞状细胞癌##恶性肿瘤放疗##化学治疗"
  }
'''

class DataProcessor(object):
    def __init__(self, root, recall_k=200, negative_sample=3):
        self.task_data_dir = os.path.join(root, 'RAW_DATA')
        self.train_path = os.path.join(self.task_data_dir, 'train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'test.json')
        self.test_path = os.path.join(self.task_data_dir, 'test.json')

        self.label_path = os.path.join(self.task_data_dir, '国际疾病分类 ICD-10北京临床版v601.xlsx')
        self.label2id, self.id2label = self._get_labels()

        self.recall_k = recall_k
        self.negative_sample = negative_sample

        self.dictionary = None
        self.index = None
        self.tfidf = None
        self.dictionary, self.index, self.tfidf = self._init_label_embedding()

        self.num_labels_cls = 2
        self.num_labels_num = 3

        self.recall = None

    def get_train_sample(self, dtype='cls', do_augment=1):
        """
        do_augment: data augment
        """
        samples = self._pre_process(self.train_path, is_predict=False)
        if dtype == 'cls':
            outputs, recall_orig_samples, recall_orig_samples_scores = self._get_cls_samples(orig_samples=samples, mode='train', do_augment=do_augment)
            return outputs, recall_orig_samples, recall_orig_samples_scores
        else:
            outputs = self._get_num_samples(orig_sample=samples, is_predict=False)
        return outputs

    def get_dev_sample(self, dtype='cls', do_augment=1):
        samples = self._pre_process(self.dev_path, is_predict=False)
        if dtype == 'cls':
            outputs, recall_orig_samples, recall_orig_samples_scores = self._get_cls_samples(orig_samples=samples, mode='eval', do_augment=do_augment)
            return outputs, recall_orig_samples, recall_orig_samples_scores
        else:
            outputs = self._get_num_samples(orig_sample=samples, is_predict=False)
        return outputs

    def get_test_sample(self, dtype='cls'):
        samples = self._pre_process(self.test_path, is_predict=True)
        if dtype == 'cls':
            outputs, recall_orig_samples, recall_orig_samples_scores = self._get_cls_samples(orig_samples=samples, mode='test')
            return outputs, recall_orig_samples, recall_orig_samples_scores
        else:
            outputs = self._get_num_samples(orig_sample=samples, is_predict=True)
        return outputs

    def get_test_orig_text(self):
        samples = load_json(self.test_path)
        texts = [sample['text'] for sample in samples]
        return texts

    def _pre_process(self, path, is_predict=False):

        samples = load_json(path)
        outputs = {'text': [], 'label': []}

        for sample in samples:
            text = self._process_single_sentence(sample['text'], mode="text")
            if is_predict:
                outputs['text'].append(text)
            else:
                label = self._process_single_sentence(sample['normalized_result'], mode="label")
                outputs['label'].append([label_ for label_ in label.split("##")])
                outputs['text'].append(text)
        return outputs

    def _save_cache(self, outputs, recall_orig_samples, mode='train'):
        cache_df = pd.DataFrame(outputs)
        cache_df.to_csv(os.path.join(self.task_data_dir, f'{mode}_samples.csv'), index=False)
        recall_orig_cache_df = pd.DataFrame(recall_orig_samples)
        recall_orig_cache_df['label'] = recall_orig_cache_df.label.apply(lambda x: " ".join([str(i) for i in x]))
        recall_orig_cache_df['recall_label'] = recall_orig_cache_df.recall_label.apply(
            lambda x: " ".join([str(i) for i in x]))
        recall_orig_cache_df.to_csv(os.path.join(self.task_data_dir, f'{mode}_recall_orig_samples.csv'),
                                    index=False)

    def _load_cache(self, mode='train'):
        outputs = {'text1': [], 'text2': [], 'label': []}
        recall_orig_samples = {'text': [], 'label': [], 'recall_label': []}

        train_cache_df = pd.read_csv(os.path.join(self.task_data_dir, f'{mode}_samples.csv'))
        outputs['text1'] = train_cache_df['text1'].values.tolist()
        outputs['text2'] = train_cache_df['text2'].values.tolist()
        outputs['label'] = train_cache_df['label'].values.tolist()

        train_recall_orig_cache_df = pd.read_csv(os.path.join(self.task_data_dir, f'{mode}_recall_orig_samples.csv'))
        recall_orig_samples['text'] = train_recall_orig_cache_df['text'].values.tolist()
        recall_orig_samples['label'] = train_recall_orig_cache_df['label'].values.tolist()
        recall_orig_samples['recall_label'] = train_recall_orig_cache_df['recall_label'].values.tolist()
        recall_orig_samples['label'] = [[int(label) for label in str(recall_orig_samples['label'][i]).split()] for i in
                                        range(len(recall_orig_samples['label']))]
        recall_orig_samples['recall_label'] = [[int(label) for label in str(recall_orig_samples['recall_label'][i]).split()] for i in
                                               range(len(recall_orig_samples['recall_label']))]
        recall_samples_scores = np.load(os.path.join(self.task_data_dir, f'{mode}_recall_score.npy'))

        return outputs, recall_orig_samples, recall_samples_scores

    def _get_cls_samples(self, orig_samples, mode='train', do_augment=1):
        if os.path.exists(os.path.join(self.task_data_dir, f'{mode}_samples.csv')) and \
                os.path.exists(os.path.join(self.task_data_dir, f'{mode}_recall_score.npy')) and \
                os.path.exists(os.path.join(self.task_data_dir, f'{mode}_recall_orig_samples.csv')):
            outputs, recall_orig_samples, recall_samples_scores = self._load_cache(mode=mode)
            return outputs, recall_orig_samples, recall_samples_scores

        outputs = {'text1': [], 'text2': [], 'label': []}

        texts = orig_samples['text']
        recall_samples_idx, recall_samples_scores = self._recall(texts)
        np.save(os.path.join(self.task_data_dir, f'{mode}_recall_score.npy'), recall_samples_scores)
        recall_orig_samples = {'text': [], 'label': [], 'recall_label': []}

        if mode == 'train':
            labels = orig_samples['label']
            for i in range(do_augment):
                for text, label in zip(texts, labels):
                    for label_ in label:
                        outputs['text1'].append(text)
                        outputs['text2'].append(label_)
                        outputs['label'].append(1)

            for text, orig_label, recall_label in zip(texts, labels, recall_samples_idx):
                orig_label_ids = [self.label2id[label] for label in orig_label]
                cnt_label = 0

                recall_orig_samples['text'].append(text)
                recall_orig_samples['label'].append(orig_label_ids)
                recall_orig_samples['recall_label'].append(recall_label)

                cur_idx = 0
                for label_ in recall_label:
                    if cnt_label >= self.negative_sample:
                        break
                    if label_ not in orig_label_ids:
                        outputs['text1'].append(text)
                        outputs['text2'].append(self.id2label[label_])
                        outputs['label'].append(0)
                        orig_label_ids.append(label_)
                        cnt_label += 1
                    cur_idx += 1
                cnt_label = 0
                recall_label = np.random.permutation(recall_label[cur_idx:])
                for label_ in recall_label:
                    if cnt_label >= self.negative_sample:
                        break
                    if label_ not in orig_label_ids:
                        outputs['text1'].append(text)
                        outputs['text2'].append(self.id2label[label_])
                        outputs['label'].append(0)
                        orig_label_ids.append(label_)
                        cnt_label += 1

            self._save_cache(outputs, recall_orig_samples, mode='train')

        elif mode == 'eval':
            labels = orig_samples['label']

            for i in range(do_augment):
                for text, label in zip(texts, labels):
                    for label_ in label:
                        outputs['text1'].append(text)
                        outputs['text2'].append(label_)
                        outputs['label'].append(1)

            for text, orig_label, recall_label in zip(texts, labels, recall_samples_idx):
                orig_label_ids = [self.label2id[label] for label in orig_label]
                recall_orig_samples['text'].append(text)
                recall_orig_samples['recall_label'].append(recall_label)
                recall_orig_samples['label'].append(orig_label_ids)

                cnt_label = 0
                cur_idx = 0
                for label_ in recall_label:
                    if cnt_label >= self.negative_sample:
                        break
                    if label_ not in orig_label_ids:
                        outputs['text1'].append(text)
                        outputs['text2'].append(self.id2label[label_])
                        outputs['label'].append(0)
                        orig_label_ids.append(label_)
                        cnt_label += 1
                    cur_idx += 1

                cnt_label = 0
                recall_label = np.random.permutation(recall_label[cur_idx:])
                for label_ in recall_label:
                    if cnt_label >= self.negative_sample:
                        break
                    if label_ not in orig_label_ids:
                        outputs['text1'].append(text)
                        outputs['text2'].append(self.id2label[label_])
                        outputs['label'].append(0)
                        orig_label_ids.append(label_)
                        cnt_label += 1

            self._save_cache(outputs, recall_orig_samples, mode='eval')

        else:
            for text, recall_label in zip(texts, recall_samples_idx):

                recall_orig_samples['text'].append(text)
                recall_orig_samples['recall_label'].append(recall_label)
                recall_orig_samples['label'].append([0])

                for label_ in recall_label:
                    outputs['text1'].append(text)
                    outputs['text2'].append(self.id2label[label_])
                    outputs['label'].append(0)
            self._save_cache(outputs, recall_orig_samples, mode='test')

        return outputs, recall_orig_samples, recall_samples_scores

    def _get_num_samples(self, orig_sample, is_predict=False):
        outputs = {'text1': [], 'text2': [], 'label': []}

        if not is_predict:
            texts = orig_sample['text']
            labels = orig_sample['label']

            for text, label in zip(texts, labels):
                outputs['text1'].append(text)
                num_labels = len(label)
                if num_labels > 2:
                    num_labels = 3
                outputs['label'].append(num_labels-1)
        else:
            outputs['text1'] = orig_sample['text']

        return outputs

    def _init_label_embedding(self):
        all_label_list = []
        for idx in range(len(self.label2id.keys())):
            all_label_list.append(list(jieba.cut(self.id2label[idx])))

        dictionary = corpora.Dictionary(all_label_list)  # 词典
        corpus = [dictionary.doc2bow(doc) for doc in all_label_list]  # 语料库
        tfidf = models.TfidfModel(corpus)  # 建立模型
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))

        return dictionary, index, tfidf

    def _recall(self, texts):
        recall_scores_idx = np.zeros((len(texts), self.recall_k), dtype=np.int)

        recall_scores = np.zeros((len(texts), self.recall_k))
        for i, x in tqdm(enumerate(texts), total=len(texts)):
            x_split = list(jieba.cut(x))
            x_vec = self.dictionary.doc2bow(x_split)
            x_sim = self.index[self.tfidf[x_vec]]  # 相似度分数 (1, labels)

            x_dices = np.zeros(len(self.label2id.keys()))
            x_set = set(x)

            for j, y in enumerate(self.label2id.keys()):
                y_set = set(y)
                x_dices[j] = len(x_set & y_set) / min(len(x_set), len(y_set))

            x_scores = x_sim + x_dices
            x_scores_idx = np.argsort(x_scores)[:len(x_scores) - self.recall_k - 1:-1]  # 由大到小排序,取前K个
            x_scores = np.sort(x_scores)[:len(x_scores) - self.recall_k - 1:-1]
            recall_scores[i] += x_scores
            recall_scores_idx[i] += x_scores_idx
        return recall_scores_idx, recall_scores

    def _get_labels(self):
        df = pd.read_excel(self.label_path, header=None)
        normalized_word = df[1].unique().tolist()
        label2id = {word: idx for idx, word in enumerate(normalized_word)}
        id2label = {idx: word for idx, word in enumerate(normalized_word)}

        num_label = len(label2id.keys())
        samples = self._pre_process(self.train_path)
        for labels in samples['label']:
            for label in labels:
                if label not in label2id:
                    label2id[label] = num_label
                    id2label[num_label] = label
                    num_label += 1

        samples = self._pre_process(self.dev_path)
        for labels in samples['label']:
            for label in labels:
                if label not in label2id:
                    label2id[label] = num_label
                    id2label[num_label] = label
                    num_label += 1
        return label2id, id2label

    def _process_single_sentence(self, sentence, mode='text'):
        sentence = str_q2b(sentence)
        sentence = sentence.strip('"')
        if mode == "text":
            sentence = sentence.replace("\\", ";")
            sentence = sentence.replace(",", ";")
            sentence = sentence.replace("、", ";")
            sentence = sentence.replace("?", ";")
            sentence = sentence.replace(":", ";")
            sentence = sentence.replace(".", ";")
            sentence = sentence.replace("/", ";")
            sentence = sentence.replace("~", "-")
        return sentence
class Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            mode='train',
            dtype='cls'
    ):
        super(Dataset, self).__init__()

        self.text1 = samples['text1']

        if dtype == 'cls':
            self.text2 = samples['text2']
            if mode != 'test':
                self.label = samples['label']
        else:
            if mode != 'test':
                self.label = samples['label']

        self.data_processor = data_processor
        self.dtype = dtype
        self.mode = mode

    def __getitem__(self, item):
        if self.dtype == 'cls':
            if self.mode != 'test':
                return self.text1[item], self.text2[item], self.label[item]
            else:
                return self.text1[item], self.text2[item]
        else:
            if self.mode != 'test':
                return self.text1[item], self.label[item]
            else:
                return self.text1[item]

    def __len__(self):
        return len(self.text1)

if __name__=="__main__":
    data_pro = DataProcessor("./")
