import numpy as np
from dataset import bulid_dataset, Dataset_BERT, build_iterator
from my_bertmodel import Config, PromptBERT
from transformers import AdamW
import torch
from utils import get_time_dif
import time
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

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, batch in enumerate(train_iter): # token_ids, label, seq_len, mask
            (input_ids, seq_len, mask), label = batch
            # print(label.shape) # ([4, 1])
            label = torch.squeeze(label)
            outputs = model(input_x=batch)

            logits = outputs.logits.argmax(dim=1)
            model.zero_grad()
            # loss_fn = nn.CrossEntropyLoss()
            # loss = loss_fn(logits.view(-1, logits.shape[2]), label.view(-1))
            loss = outputs.loss
            total_loss += loss
            loss.backward()
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



def main():
    # train()

    model = PromptBERT()
    config = Config()
    model.to(config.device)
    # text = '臭男人。'
    # test_sentence(config, model, text)

    test_data = bulid_dataset(config, type='test')
    test_iter = build_iterator(test_data, config)
    test(config, model, test_iter)
    # # ['无感', '振奋', '厌恶', '惊喜', '开心', '害怕', '难过', '生气']

if __name__ == '__main__':
    main()
