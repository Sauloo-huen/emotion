from config.config import path, templete
from transformers import BertForMaskedLM, BertTokenizer
import torch
import torch.nn as nn


class Config(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'Prompt-BERT'
        # self.train_path = r'D:\Python_projection\Bert-Chinese-Prompt-Mask-main\data\Train.txt'  # 训练集
        # self.test_path = dataset + '/data/test.txt'  # 测试集 data/Train.txt
        self.save_path = f'D:\Python_projection\Bert-Chinese-Prompt-Mask-main\saved_dict\{self.model_name}.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.target_names = ['无感', '振奋', '厌恶', '惊喜', '开心', '害怕', '难过', '生气']
        self.require_improvement = 5000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 8 # 类别数 8
        self.num_epochs = 5  # epoch数
        self.batch_size = 4  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5  # 学习率
        self.bert_path = 'bert-base-chinese'  # bert预训练模型位置
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # bert切分词
        self.hidden_size = 768  # bert隐藏层个数

# inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
#
# with torch.no_grad():
#     logits = model(**inputs).logits
#
# # retrieve index of [MASK]
# mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
#
# predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
# tokenizer.decode(predicted_token_id)

class PromptBERT(nn.Module):
    def __init__(self):
        super(PromptBERT, self).__init__()
        self.roberta = BertForMaskedLM.from_pretrained('bert-base-chinese')
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def forward(self, input_x, device): # (x, seq_len, mask), y
        (input_ids, seq_len, mask), label, mask_token_index = input_x
        # print(input_ids.shape)
        # print(1)
        # mask_token_index = (token_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        outputs = self.roberta(input_ids=input_ids, attention_mask=mask)

        # logits = outputs.logits
        # predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        # x = self.tokenizer.decode(predicted_token_id)
        return outputs