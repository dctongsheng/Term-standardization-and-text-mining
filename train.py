# coding=utf-8
import os
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

from utils import seed_everything, ProgressBar, TokenRematch

from sklearn.metrics import precision_recall_fscore_support


def cls_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')


def num_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')
def commit_prediction(text, preds, num_preds, recall_labels, recall_scores, output_dir, id2label):
    text1 = text

    pred_result = []
    active_indices = (preds >= 0.5)
    for text, active_indice, pred, num, recall_label, recall_score in zip(text1, active_indices, preds, num_preds, recall_labels, recall_scores):
        tmp_dict = {'text': text, 'normalized_result': []}

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

    with open(os.path.join(output_dir, 'CHIP-CDN_test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))


class Trainer(object):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):

        self.args = args
        self.model = model
        self.data_processor = data_processor
        self.tokenizer = tokenizer

        if train_dataset is not None and isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset

        if eval_dataset is not None and isinstance(eval_dataset, Dataset):
            self.eval_dataset = eval_dataset

        self.logger = logger
        self.model_class = model_class
        self.ngram_dict = ngram_dict

    def train(self):
        args = self.args
        logger = self.logger
        model = self.model
        model.to(args.device)

        train_dataloader = self.get_train_dataloader()

        num_training_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = num_training_steps * args.warmup_proportion
        num_examples = len(train_dataloader.dataset)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)

        if args.task_name in ['qic', 'qqr', 'qtr', 'sts']:
            seed_everything(args.seed)
            model.zero_grad()

        logger.info("***** Running training *****")
        logger.info("Num samples %d", num_examples)
        logger.info("Num epochs %d", args.epochs)
        logger.info("Num training steps %d", num_training_steps)
        logger.info("Num warmup steps %d", num_warmup_steps)

        global_step = 0
        best_step = None
        best_score = .0
        cnt_patience = 0
        for i in range(args.epochs):
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            for step, item in enumerate(train_dataloader):
                loss = self.training_step(model, item)
                pbar(step, {'loss': loss.item()})

                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                if args.task_name in ['qic', 'qqr', 'qtr', 'sts']:
                    model.zero_grad()
                else:
                    optimizer.zero_grad()

                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print("")
                    score = self.evaluate(model)
                    if score > best_score:
                        best_score = score
                        best_step = global_step
                        cnt_patience = 0
                        self._save_checkpoint(model, global_step)
                    else:
                        cnt_patience += 1
                        self.logger.info("Earlystopper counter: %s out of %s", cnt_patience, args.earlystop_patience)
                        if cnt_patience >= self.args.earlystop_patience:
                            break
            if cnt_patience >= args.earlystop_patience:
                break

        logger.info("Training Stop! The best step %s: %s", best_step, best_score)
        if args.device == 'cuda':
            torch.cuda.empty_cache()

        self._save_best_checkpoint(best_step=best_step)

        return global_step, best_step

    def evaluate(self, model):
        raise NotImplementedError

    def _save_checkpoint(self, model, step):
        raise NotImplementedError

    def _save_best_checkpoint(self, best_step):
        raise NotImplementedError

    def training_step(self, model, item):
        raise NotImplementedError

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True
        )

    def get_eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False
        )

    def get_test_dataloader(self, test_dataset, batch_size=None):
        if not batch_size:
            batch_size = self.args.eval_batch_size

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )



class CLSTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            recall_orig_eval_samples=None,
            recall_orig_eval_samples_scores=None,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(CLSTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

        self.recall_orig_eval_samples = recall_orig_eval_samples
        self.recall_orig_eval_samples_scores = recall_orig_eval_samples_scores

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        text2 = item[1]
        labels = item[2].to(self.args.device)


        inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                    truncation='longest_first', max_length=self.args.max_length)

        inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

        if self.args.model_type == 'zen':
            inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
            inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
            inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
            inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

        outputs = model(labels=labels, **inputs)
        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluation')
        for step, item in enumerate(eval_dataloader):
            model.eval()

            text1 = item[0]
            text2 = item[1]
            label = item[2].to(args.device)


            inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                        truncation='longest_first', max_length=self.args.max_length)

            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs

            if preds is None:
                preds = logits.detach().cpu().numpy()
                labels = label.cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu(), axis=0)
                labels = np.append(labels, label.detach().cpu().numpy(), axis=0)

            pbar(step, info="")

        preds = np.argmax(preds, axis=1)

        p, r, f1, _ = cls_metric(preds, labels)
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return f1

    def predict(self, test_dataset, model):
        args = self.args
        logger = self.logger
        test_dataset.text1 = test_dataset.text1
        test_dataset.text2 = test_dataset.text2
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        preds = None

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Evaluation')
        for step, item in enumerate(test_dataloader):
            model.eval()

            text1 = item[0]
            text2 = item[1]


            inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                        truncation='longest_first', max_length=self.args.max_length)

            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs

            if preds is None:
                preds = logits.detach().softmax(-1)[:, 1].cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().softmax(-1)[:, 1].cpu().numpy(), axis=0)

            pbar(step, info="")

        preds = preds.reshape(len(preds) // args.recall_k, args.recall_k)
        np.save(os.path.join(args.result_output_dir, f'cdn_test_preds.npy'), preds)
        return preds

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.pt'))
        self.logger.info('Saving models checkpoint to %s', output_dir)

        model.encoder.save_pretrained(output_dir)
        self.tokenizer.save_vocabulary(save_directory=output_dir)

    def _save_best_checkpoint(self, best_step):
        pass


class NUMTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(NUMTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        labels = item[1].to(self.args.device)


        inputs = self.tokenizer(text1, padding='max_length', max_length=self.args.max_length,
                                    truncation=True, return_tensors='pt')

        inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

        if self.args.model_type == 'zen':
            inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
            inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
            inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
            inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

        outputs = model(labels=labels, **inputs)
        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            text1 = item[0]
            labels = item[1].to(args.device)

            inputs = self.tokenizer(text1, padding='max_length', max_length=self.args.max_length,
                                        truncation=True, return_tensors='pt')
            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(labels=labels, **inputs)
                loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, labels.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        p, r, f1, _ = num_metric(preds, eval_labels)
        logger.info("%s-%s f1: %s", args.task_name, args.model_name, f1)
        return f1

    def predict(self, model, test_dataset, orig_texts, cls_preds, recall_labels, recall_scores):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        preds = None

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Evaluation')
        for step, item in enumerate(test_dataloader):
            model.eval()

            text1 = item


            inputs = self.tokenizer(text1, padding='max_length', max_length=self.args.max_length,
                                        truncation=True, return_tensors='pt')

            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)

                if self.args.model_type == 'zen':
                    logits = outputs
                else:
                    logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

            pbar(step, info="")
        preds = np.argmax(preds, axis=1)

        recall_labels = np.array(recall_labels['recall_label'])
        recall_scores = recall_scores
        commit_prediction(orig_texts, cls_preds, preds, recall_labels, recall_scores,
                              args.result_output_dir, self.data_processor.id2label)

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        self.tokenizer.save_vocabulary(save_directory=output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)

    def _save_best_checkpoint(self, best_step):
        model = self.model_class.from_pretrained(os.path.join(self.args.output_dir, f'checkpoint-{best_step}'),
                                                 num_labels=self.data_processor.num_labels_num)
        if not os.path.exists(os.path.join(self.args.output_dir, 'num')):
            os.mkdir(os.path.join(self.args.output_dir, 'num'))
        model.save_pretrained(os.path.join(self.args.output_dir, 'num'))
        torch.save(self.args, os.path.join(os.path.join(self.args.output_dir, 'num'), 'training_args.bin'))
        self.tokenizer.save_vocabulary(save_directory=os.path.join(self.args.output_dir, 'num'))
        self.logger.info('Saving models checkpoint to %s', os.path.join(self.args.output_dir, 'num'))