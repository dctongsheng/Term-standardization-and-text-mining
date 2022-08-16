# coding=utf-8

'''
scp -r ./医疗术语标准化 wsbi@192.168.0.74:/home/wsbi/workspace/nlp_yiliao

'''
import os
import sys
sys.path.append('.')
import argparse
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, AlbertModel, BertForSequenceClassification, \
    AlbertForSequenceClassification
import transformers
transformers.logging.set_verbosity_error()
from model import CLSModel
from train import CLSTrainer, NUMTrainer
from utils import init_logger, seed_everything
from data import Dataset, DataProcessor

"hfl/chinese-bert-wwm"
argg={'data_dir': './', 'model_dir': 'hfl', 'model_type': 'bert', 'model_name': 'chinese-bert-wwm-ext', 'task_name': 'cdn', 'output_dir': './data/output/cdn/', 'do_train': True, 'do_predict': False, 'result_output_dir': './data/result', 'max_length': 64,
      'train_batch_size': 32, 'eval_batch_size': 128, 'learning_rate': 3e-05, 'weight_decay': 0.01, 'adam_epsilon': 1e-08, 'max_grad_norm': 0.0, 'epochs': 3, 'warmup_proportion': 0.1, 'earlystop_patience': 10, 'logging_steps': 300, 'save_steps': 300, 'seed': 2021,
      'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),'recall_k':200,'num_neg':5,'do_aug':6}

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


args = Dict(argg)

logger=init_logger("logger.log")

tokenizer_class, model_class = (BertTokenizer, BertModel)
logger.info('Training CLS model...')
tokenizer = tokenizer_class.from_pretrained("hfl/chinese-bert-wwm")

ngram_dict = None

data_processor = DataProcessor(root=args.data_dir, recall_k=args.recall_k,
                                  negative_sample=args.num_neg)
train_samples, recall_orig_train_samples, recall_orig_train_samples_scores = data_processor.get_train_sample(
    dtype='cls', do_augment=args.do_aug)
eval_samples, recall_orig_eval_samples, recall_orig_train_samples_scores = data_processor.get_dev_sample(dtype='cls',
                                                                                                         do_augment=args.do_aug)
if data_processor.recall:
    logger.info('first recall score: %s', data_processor.recall)

train_dataset = Dataset(train_samples, data_processor, dtype='cls', mode='train')
eval_dataset = Dataset(eval_samples, data_processor, dtype='cls', mode='eval')

model = CLSModel(model_class, encoder_path="hfl/chinese-bert-wwm",
                       num_labels=data_processor.num_labels_cls)
cls_model_class = BertForSequenceClassification
trainer = CLSTrainer(args=args, model=model, data_processor=data_processor,
                           tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                           logger=logger, recall_orig_eval_samples=recall_orig_eval_samples,
                           model_class=cls_model_class,
                           recall_orig_eval_samples_scores=recall_orig_train_samples_scores,
                           ngram_dict=ngram_dict)

global_step, best_step = trainer.train()

model = CLSModel(model_class, encoder_path=os.path.join(args.output_dir, f'checkpoint-{best_step}'),
                       num_labels=data_processor.num_labels_cls)
model.load_state_dict(torch.load(os.path.join(args.output_dir, f'checkpoint-{best_step}', 'pytorch_model.pt')))
tokenizer = tokenizer_class.from_pretrained(os.path.join(args.output_dir, f'checkpoint-{best_step}'))
torch.save(model.state_dict(), os.path.join(args.output_dir, 'pytorch_model_cls.pt'))
if not os.path.exists(os.path.join(args.output_dir, 'cls')):
    os.mkdir(os.path.join(args.output_dir, 'cls'))

model.encoder.save_pretrained(os.path.join(args.output_dir, 'cls'))

tokenizer.save_vocabulary(save_directory=os.path.join(args.output_dir, 'cls'))
logger.info('Saving models checkpoint to %s', os.path.join(args.output_dir, 'cls'))

logger.info('Training NUM model...')
args.logging_steps = 30
args.save_steps = 30
train_samples = data_processor.get_train_sample(dtype='num', do_augment=1)
eval_samples = data_processor.get_dev_sample(dtype='num')
train_dataset = Dataset(train_samples, data_processor, dtype='num', mode='train')
eval_dataset = Dataset(eval_samples, data_processor, dtype='num', mode='eval')

cls_model_class = BertForSequenceClassification
model = cls_model_class.from_pretrained("hfl/chinese-bert-wwm",
                                        num_labels=data_processor.num_labels_num)
trainer = NUMTrainer(args=args, model=model, data_processor=data_processor,
                           tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                           logger=logger, model_class=cls_model_class, ngram_dict=ngram_dict)

global_step, best_step = trainer.train()
