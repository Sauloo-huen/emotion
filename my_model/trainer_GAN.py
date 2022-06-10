import numpy as np
from dataset import bulid_dataset, Dataset_BERT, build_iterator
from my_bertmodel import Config, PromptBERT
from transformers import AdamW
import torch
from sklearn import metrics
from utils import get_time_dif
import time
import torch.nn.functional as F
import torch.nn as nn

from transformers import logging

logging.set_verbosity_error() # cancel warning


def train():
    model = PromptBERT()
    config = Config()
    model.to(config.device)

    train_data = bulid_dataset(config, type='train')
    train_iter = build_iterator(train_data, config)
    dev_data = bulid_dataset(config, type='dev')
    dev_iter = build_iterator(dev_data, config)
    test_data = bulid_dataset(config, type='test')
    test_iter = build_iterator(test_data, config)

    dev_best_loss = float('inf')
    start_time = time.time()
    total_batch = 0  # 记录进行到多少batch
    last_improve = 0  # 记录上次验证集loss下降的batch数
    total_loss = 0
    flag = False  # 记录是否很久没有效果提升
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    model.train()

    # GAN
    loss_func0 = nn.CrossEntropyLoss()
    loss_map = {"0": loss_func0}
    smart_adv = SmartPerturbation(model=model, loss_map=loss_map)
    adv_alpha = 0.01

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, batch in enumerate(train_iter): # token_ids, label, seq_len, mask
            (input_ids, seq_len, mask), label = batch
            # print(label.shape) # ([4, 1])
            label = torch.squeeze(label)
            outputs = model(input_x=batch)
            logit_before = outputs.logits
            logits = outputs.logits.argmax(dim=1)
            model.zero_grad()
            # loss_fn = nn.CrossEntropyLoss()
            # loss = loss_fn(logits.view(-1, logits.shape[2]), label.view(-1))
            loss = outputs.loss
            adv_loss = smart_adv.forward(logit_before, batch)
            loss_sum = adv_alpha*adv_loss + loss
            total_loss += loss_sum
            loss_sum.backward()
            optimizer.step()

            predicted_token_id = logits.cpu().numpy()
            label = label.cpu().numpy()
            predict_all = np.append(predict_all, predicted_token_id)
            labels_all = np.append(labels_all, label)

            if i % 100 == 0:
                acc = 0
                acc = ((predict_all == labels_all).sum()).item()
                # for i in range(len(labels_all)):
                #     acc += 1 if all(labels_all[i] == predict_all[i]) else 0
                train_acc = acc / len(labels_all)

                predict_all = np.array([], dtype=int)
                labels_all = np.array([], dtype=int) # 重置
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}, Val Loss: {3:>5.2}, Best Val Loss: {4:>5.2},  Val Acc: {5:>6.2%},  Time: {6} {7}'
                print(msg.format(total_batch, total_loss/100, train_acc, dev_loss, dev_best_loss, dev_acc, time_dif, improve))
                model.train()
                total_loss = 0
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss = evaluate(config, model, test_iter)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def test_sentence(config, model, text):
    data = []
    mask_token_index = []
    model.load_state_dict(torch.load(config.save_path))
    token = '[CLS] ' + text + ' [SEP] [SEP] ' + '这真让人[MASK][MASK]。' + ' [SEP]'
    token = config.tokenizer.tokenize(token)
    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    mask_token_ids = token_ids.index(config.tokenizer.mask_token_id)
    mask_token_index.append(mask_token_ids)
    mask_token_index.append(mask_token_ids + 1)
    mask = [1] * len(token_ids)
    label = config.tokenizer.convert_tokens_to_ids('开心')
    data.append((token_ids, label, len(token_ids), mask, mask_token_index))
    x = torch.LongTensor([_[0] for _ in data]).to(config.device)  # _[0]即每个batch里面的Token_ids
    y = torch.LongTensor([_[1] for _ in data]).to(config.device)
    # pad前的长度(超过pad_size的设为pad_size)
    seq_len = torch.LongTensor([_[2] for _ in data]).to(config.device)
    mask = torch.LongTensor([_[3] for _ in data]).to(config.device)
    mask_token_index = torch.LongTensor([_[4] for _ in data]).to(config.device)
    datas = (x, seq_len, mask), y, mask_token_index

    outputs = model(datas, config.device)
    logits = outputs.logits
    mask_token = (x == 103)
    predicted_token = logits[mask_token]
    predicted_token_id = predicted_token.argmax(axis=-1)
    predic_label = config.tokenizer.decode(predicted_token_id)
    print(predic_label)

def evaluate(config, model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_iter:
            (input_ids, seq_len, mask), label = batch
            # print(label.shape) # ([4, 1])
            label = torch.squeeze(label)
            outputs = model(input_x=batch)
            logits = outputs.logits.argmax(dim=1)
            model.zero_grad()
            # loss_fn = nn.CrossEntropyLoss()
            # loss = loss_fn(logits.view(-1, logits.shape[2]), label.view(-1))
            loss = outputs.loss
            loss_total += loss
            predicted_token_id = logits.cpu().numpy()
            label = label.cpu().numpy()
            predict_all = np.append(predict_all, predicted_token_id)
            labels_all = np.append(labels_all, label)

    # acc = 0
    # for i in range(len(labels_all)):
    #     acc += 1 if all(labels_all[i] == predict_all[i]) else 0
    acc = ((predict_all == labels_all).sum()).item()
    acc = acc / len(labels_all)


    return acc, loss_total / len(data_iter)

class SmartPerturbation():
    """
    step_size noise扰动学习率
    epsilon 梯度scale时防止分母为0
    norm_p 梯度scale采用的范式
    noise_var 扰动初始化系数
    loss_map 字典，loss函数的类型{"0":mse(),....}
    使用方法
    optimizer =
    model =
    loss_func =
    loss_map = {"0":loss_fun0,"1":loss_fun1,...}
    smart_adv = SmartPerturbation(model,epsilon,step_size,noise_var,loss_map)
    for batch_input, batch_label in data:
        inputs = {'input_ids':...,...,'labels':batch_label}
        logits = model(**inputs)
        loss = loss_func(logits,batch_label)
        loss_adv = smart_adv.forward(logits)
        loss = loss + adv_alpha*loss_adv
        loss.backward()
        optimizer.step()
        model.zero_grad()
    """

    def __init__(self,
                 model,
                 epsilon=1e-7,
                 multi_gpu_on=False,
                 step_size=1e-3, # 1e-3
                 noise_var=1e-5, # 1e-5
                 norm_p='inf',
                 k=1,
                 fp16=False,
                 loss_map={},
                 norm_level=0):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta
        self.step_size = step_size
        self.multi_gpu_on = multi_gpu_on
        self.fp16 = fp16
        self.K = k
        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p
        self.model = model
        self.loss_map = loss_map
        self.norm_level = norm_level > 0
        assert len(loss_map) > 0

    # 梯度scale
    def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
        if self.norm_p == 'l2':
            if sentence_level:
                direction = grad / (torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon)
            else:
                direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + self.epsilon)
        elif self.norm_p == 'l1':
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (grad.abs().max((-2, -1), keepdim=True)[0] + self.epsilon)
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
                eff_direction = eff_grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
        return direction, eff_direction

    # 初始noise扰动
    def generate_noise(self, embed, mask, epsilon=1e-6):
        noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
        noise.detach()
        noise.requires_grad_()
        return noise

    # 对称散度loss
    def stable_kl(self, logit, target, epsilon=1e-7, reduce=True):
        # logit = logit.argmax(dim=1)
        logit = logit.view(-1, logit.size(-1)).float()
        target = target.view(-1, target.size(-1)).float()
        bs = logit.size(0)
        p = F.log_softmax(logit, 1).exp()
        y = F.log_softmax(target, 1).exp()
        rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
        ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
        if reduce:
            return (p * (rp - ry) * 2).sum() / bs
        else:
            return (p * (rp - ry) * 2).sum()

    # 对抗loss输出
    def forward(self,
                logits,
                input_x,
                task_id=0,
                task_type="Classification",
                pairwise=1):
        # adv training
        assert task_type in set(['Classification', 'Ranking', 'Regression']), 'Donot support {} yet'.format(task_type)
        (input_ids, seq_len, mask), label = input_x
        vat_args = {'input_x': input_x, 'output_attentions': True, 'output_hidden_states': True, 'use_input': True}
        # init delta
        outputs = self.model(**vat_args) # embed [B,S,h_dim] h_dim=768
        embed = outputs.hidden_states[0]
        # embed生成noise
        noise = self.generate_noise(embed, mask, epsilon=self.noise_var)
        # noise更新K轮
        for step in range(0, self.K):
            vat_args = {'inputs_embeds': embed + noise, 'use_input': False}
            # noise+embed得到对抗样本的输出logits
            adv_logits = self.model(**vat_args).logits
            adv_loss = self.stable_kl(adv_logits, logits.detach(), reduce=False)
            # adv_loss.requires_grad = True

            # 得到noise的梯度
            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad = noise + delta_grad * self.step_size
            # 得到新的scale的noise
            noise, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()
        vat_args = {'inputs_embeds': embed + noise, 'use_input': False}
        adv_logits = self.model(**vat_args).logits
        if task_type == 'Ranking':
            adv_logits = adv_logits.view(-1, pairwise)
        # adv_lc = self.loss_map[task_id] # loss_fn
        # 计算对抗样本的对抗损失
        # adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        adv_loss = loss_fn(logits, adv_logits)
        return adv_loss

def main():
    train()
    # model = PromptBERT()
    # config = Config()
    # model.to(config.device)
    # text = '臭男人。'
    # test_sentence(config, model, text)
    # # ['无感', '振奋', '厌恶', '惊喜', '开心', '害怕', '难过', '生气']

if __name__ == '__main__':
    main()
