import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging
from transformers import BertTokenizer, BertForSequenceClassification
import json
import jsonlines
task_to_keys = {
    "sst2": ("sentence", None)
}

logger = logging.getLogger(__name__)


def bulid_data(tokenizer, path, pad_size=30):
    data = []
    tokenizer = tokenizer
    f = open(path, 'r', encoding='utf-8')
    for line in jsonlines.Reader(f):
        text=line['text']
        label=line['label']
        text_tokenize = tokenizer.tokenize(text)
        text_tokenize = tokenizer.convert_tokens_to_ids(text_tokenize)
        mask = []
        if pad_size:
            if len(text_tokenize) < pad_size:
                mask = [1] * len(text_tokenize) + [0] * (pad_size - len(text_tokenize))
                text_tokenize += ([0] * (pad_size - len(text_tokenize)))
            else:
                mask = [1] * pad_size
                text_tokenize = text_tokenize[:pad_size]
                # print((text_tokenize, int(label), mask))
        data.append((text_tokenize, int(label), mask))
    print(f'{len(data)}data has load!')
    # print(data)
    return data

class Dataset_BERT(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0: # 不能被整除，即最后一个batch不够batch_size
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # print('here:_to_tensor')
        # token_ids, label, seq_len, mask
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device) # _[0]即每个batch里面的Token_ids
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # print((x, mask), y)
        return (x, mask), y

    def __next__(self):
        # print('here:__next__')
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)

            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)

            return batches


    def __iter__(self):
        # print('here:__iter__')
        return self

    def __len__(self):
        if self.residue: # 不能被整除时
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = Dataset_BERT(dataset, config.batch_size, config.device)
    print(iter)
    return iter


#
# class GlueDataset():
#     def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
#         super().__init__()
#         # raw_datasets = load_dataset("glue", data_args.dataset_name)
#         raw_datasets = '/common-data/new_build/xiaoying.huang/emotion-main/my_model1/tasks/glue/sst2'
#         self.tokenizer = tokenizer
#         self.data_args = data_args
#         #labels
#         self.is_regression = data_args.dataset_name == "stsb"
#         if not self.is_regression:
#             self.label_list = raw_datasets["train"].features["label"].names
#             self.num_labels = len(self.label_list)
#         else:
#             self.num_labels = 1
#
#
#
#         # Preprocessing the raw_datasets
#         self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]
#
#         # Padding strategy
#         if data_args.pad_to_max_length:
#             self.padding = "max_length"
#         else:
#             # We will pad later, dynamically at batch creation, to the max sequence length in each batch
#             self.padding = False
#
#         # Some models have set the order of the labels to use, so let's make sure we do use it.
#         if not self.is_regression:
#             self.label2id = {l: i for i, l in enumerate(self.label_list)}
#             self.id2label = {id: label for label, id in self.label2id.items()}
#
#         if data_args.max_seq_length > tokenizer.model_max_length:
#             logger.warning(
#                 f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
#                 f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
#             )
#         self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
#
#         raw_datasets = raw_datasets.map(
#             self.preprocess_function,
#             batched=True,
#             load_from_cache_file=not data_args.overwrite_cache,
#             desc="Running tokenizer on dataset",
#         )
#
#         if training_args.do_train:
#             self.train_dataset = raw_datasets["train"]
#             if data_args.max_train_samples is not None:
#                 self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))
#
#         if training_args.do_eval:
#             self.eval_dataset = raw_datasets["validation_matched" if data_args.dataset_name == "mnli" else "validation"]
#             if data_args.max_eval_samples is not None:
#                 self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))
#
#         if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
#             self.predict_dataset = raw_datasets["test_matched" if data_args.dataset_name == "mnli" else "test"]
#             if data_args.max_predict_samples is not None:
#                 self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))
#
#         self.metric = load_metric("glue", data_args.dataset_name)
#
#         if data_args.pad_to_max_length:
#             self.data_collator = default_data_collator
#         elif training_args.fp16:
#             self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
#
#
#     def preprocess_function(self, examples):
#         # Tokenize the texts
#         args = (
#             (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
#         )
#         result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)
#
#         return result
#
#     def compute_metrics(self, p: EvalPrediction):
#         preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
#         preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
#         if self.data_args.dataset_name is not None:
#             result = self.metric.compute(predictions=preds, references=p.label_ids)
#             if len(result) > 1:
#                 result["combined_score"] = np.mean(list(result.values())).item()
#             return result
#         elif self.is_regression:
#             return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
#         else:
#             return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

def main():
    tokenizer = BertTokenizer.from_pretrained('/common-data/new_build/xiaoying.huang/emotion-main/bert-base-uncased')
    path = '/common-data/new_build/xiaoying.huang/emotion-main/my_model1/tasks/glue/sst2/train.jsonl'
    bulid_data(tokenizer, path)

if __name__ == '__main__':
    main()
