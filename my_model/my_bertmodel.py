# from config.config import path, templete
from transformers import BertForMaskedLM, BertTokenizer
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, RobertaForSequenceClassification,RobertaTokenizer


class Config(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'gan-prompt-robert-2classes-acc'
        # self.train_path = r'D:\Python_projection\Bert-Chinese-Prompt-Mask-main\data\Train.txt'  # 训练集
        # self.test_path = dataset + '/data/test.txt'  # 测试集 data/Train.txt
        self.save_path = f'/common-data/new_build/xiaoying.huang/emotion-main/saved_dict/{self.model_name}.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 5000 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2 # 类别数 8
        self.num_epochs = 20  # epoch数
        self.batch_size = 4  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5  # 学习率
        # self.bert_path = '/common-data/new_build/xiaoying.huang/emotion-main/bert-base-uncased' # bert预训练模型位置
        self.bert_path = '/common-data/new_build/xiaoying.huang/emotion-main/roberta-large' # bert预训练模型位置
        self.tokenizer = RobertaTokenizer.from_pretrained(self.bert_path)  # bert切分词
        self.hidden_size = 768  # bert隐藏层个数


class PromptBERT(nn.Module):
    def __init__(self):
        super(PromptBERT, self).__init__()
        self.roberta = BertForSequenceClassification.from_pretrained(
            '/common-data/new_build/xiaoying.huang/emotion-main/bert-base-uncased',
                                                                    num_labels=2 ) # default num=2
        self.tokenizer = BertTokenizer.from_pretrained(
            '/common-data/new_build/xiaoying.huang/emotion-main/bert-base-uncased')

    def forward(self, input_x=None, inputs_embeds=None, output_attentions=False, output_hidden_states=False, use_input=True): # (x, seq_len, mask), y
        if use_input:
            (input_ids, seq_len, mask), label = input_x
        else:
            (input_ids, seq_len, mask), label = (None, None, None), None

        if output_hidden_states == False:
            # outputs = self.roberta(input_ids=input_ids, attention_mask=mask, labels=label)

            if use_input:
                outputs = self.roberta(input_ids=input_ids, attention_mask=mask, labels=label)
            else:
                outputs = self.roberta(attention_mask=mask, inputs_embeds=inputs_embeds, labels=label)
        else:

            if use_input:
                outputs = self.roberta(input_ids=input_ids, attention_mask=mask, inputs_embeds=inputs_embeds,
                                   output_attentions=True, output_hidden_states=True, labels=label)
            else:
                outputs = self.roberta(attention_mask=mask, inputs_embeds=inputs_embeds,
                                       output_attentions=True, output_hidden_states=True, labels=label)

        return outputs

class get_emb():
    def __init__(self):
        super(get_emb, self).__init__()
        self.device = 'cuda'
        self.pre_seq_len = 2
        self.prefix_tokens = torch.arange(self.pre_seq_len).long().to(self.device)  # 连续自动prompt都这样写
        #
        # self.model = BertForSequenceClassification.from_pretrained(
        #     r'/common-data/new_build/xiaoying.huang/emotion-main/bert-base-uncased',
        #     num_labels=2)
        # print(self.model)
        # self.emb = self.model.bert.embeddings.to(self.device)
        # self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, 768).to(self.device)  # 除了embedding 也可以加其他层

        self.model = RobertaForSequenceClassification.from_pretrained(
            r'/common-data/new_build/xiaoying.huang/emotion-main/roberta-large',
            num_labels=2)
        print(self.model)
        self.emb = self.model.roberta.embeddings.to(self.device)
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, 1024).to(self.device)  # 除了embedding 也可以加其他层
        for param in self.model.parameters():
            param.requires_grad = False
        # self.reparam = nn.Sequential(
        #     nn.Linear(self.model.bert.embeddings, self.model.hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(self.model.hidden_size, self.model.hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(self.model.hidden_size, 2 * self.model.base_config.n_layer * self.model.bert.embeddings)
        # )


    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device) # 加一维batch size
        prompts = self.prefix_encoder(prefix_tokens) # 经过embedding层，多一维hidden size
        return prompts # (batch_size, seq_len, hidden_size)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        raw_emb = self.emb(
            input_ids=input_ids
        )
        prompts = self.get_prompt(batch_size=batch_size)
        inputs_embeds = torch.cat((prompts, raw_emb), dim=1)  # 两个不同的embedding拼在一起
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.device)  # 全一矩阵表示prompt的mask，prompt里没有padding
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1) # [4,64]

        return inputs_embeds, attention_mask

class PromptBERT_new(nn.Module):
    def __init__(self):
        super(PromptBERT_new, self).__init__()
        self.roberta = BertForSequenceClassification.from_pretrained(r'/common-data/new_build/xiaoying.huang/emotion-main/bert-base-chinese',
                                                       num_labels=8)
        # self.roberta = BertForSequenceClassification.from_pretrained(
        #     r'/common-data/new_build/xiaoying.huang/emotion-main/Erlangshen-Roberta-110M-Sentiment',
        #     num_labels=8)

    def forward(self, input_emb, attention_mask, label):
        outputs = self.roberta(inputs_embeds=input_emb, attention_mask=attention_mask, labels=label)

        return outputs

class BERT_new(nn.Module):
    def __init__(self):
        super(BERT_new, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            r'/common-data/new_build/xiaoying.huang/emotion-main/roberta-large',
            num_labels=2)

    def forward(self, input_ids, attention_mask, label):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, labels=label)

        return outputs

class PromptRoberta(nn.Module):
    def __init__(self):
        super(PromptRoberta, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            r'/common-data/new_build/xiaoying.huang/emotion-main/roberta-large',
            num_labels=2)
        # self.roberta = RobertaForSequenceClassification.from_pretrained(
        #     r'/common-data/new_build/xiaoying.huang/emotion-main/Erlangshen-Roberta-330M-Similarity',
        #     num_labels=8)

    def forward(self, input_emb, attention_mask, label):
        outputs = self.roberta(inputs_embeds=input_emb, attention_mask=attention_mask, labels=label)

        return outputs


class bert_gan(nn.Module):
    def __init__(self):
        super(bert_gan, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            r'/common-data/new_build/xiaoying.huang/emotion-main/roberta-large',
            num_labels=2)
        self.emb = self.model.roberta.embeddings.to('cuda')

    def forward(self, input_emb, attention_mask, label):
        outputs = self.model(inputs_embeds=input_emb, attention_mask=attention_mask, labels=label)

        return outputs

    def get_emb(self, input_ids, attention_mask):
        raw_emb = self.emb(
            input_ids=input_ids
        )
        return raw_emb, attention_mask





