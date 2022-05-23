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

from importlib import import_module


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
    flag = False  # 记录是否很久没有效果提升
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, batch in enumerate(train_iter): # token_ids, label, seq_len, mask
            (input_ids, seq_len, mask), label, mask_token_index = batch
            # print(label.shape) # ([4, 2])
            label = label.reshape([8, 1])
            label = torch.squeeze(label)
            # print(label.shape)
            outputs = model(batch, config.device)
            logits = outputs.logits # (batch_size, sequence_length, config.vocab_size) ([4, 60, 21128])
            # logits = torch.argmax(logits, dim=-1) # ([4, 60])
            mask_token = (input_ids==103) # ([4, 60])
            # print(logits.shape)
            # print(mask_token.shape)
            predicted_token = logits[mask_token] # [6] 有两个mask因为超长度被切出去了 ([8, 21128])
            # print(predicted_token_id.shape)
            model.zero_grad()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(predicted_token, label)
            loss.backward()
            optimizer.step()

            predicted_token_id = predicted_token.argmax(axis=-1)
            predicted_token_id = predicted_token_id.reshape([4, 2]).cpu().numpy()
            label = label.reshape([4, 2]).cpu().numpy()
            predict_all = np.append(predict_all, predicted_token_id)
            labels_all = np.append(labels_all, label)

            if i % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                # true_label = config.tokenizer.decode(label)
                # predic_label = config.tokenizer.decode(predicted_token_id)
                # train_acc = metrics.accuracy_score(true_label, predic_label)

                train_acc = metrics.accuracy_score(labels_all, predict_all)
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
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_best_loss, dev_acc, time_dif, improve))
                model.train()
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
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_iter:
            (input_ids, seq_len, mask), label, mask_token_index = batch

            label = label.reshape([8, 1])
            label = torch.squeeze(label)
            # print(label.shape)
            outputs = model(batch, config.device)
            logits = outputs.logits  # (batch_size, sequence_length, config.vocab_size) ([4, 60, 21128])
            # logits = torch.argmax(logits, dim=-1) # ([4, 60])
            mask_token = (input_ids == 103)  # ([4, 60])
            # print(logits.shape)
            # print(mask_token.shape)
            predicted_token = logits[mask_token]  # [6] 有两个mask因为超长度被切出去了 ([8, 21128])
            # print(predicted_token_id.shape)
            model.zero_grad()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(predicted_token, label)

            loss_total += loss
            predicted_token_id = predicted_token.argmax(axis=-1)
            predic_label = config.tokenizer.decode(predicted_token_id) # [8]
            predic_label = predic_label.reshape([4,2])
            true_label = config.tokenizer.decode(label) # [8]
            true_label = true_label.reshape([4,2])
            # labels = labels.data.cpu().numpy()
            # predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, true_label)
            predict_all = np.append(predict_all, predic_label)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.target_names, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

def main():
    train()

if __name__ == '__main__':
    main()

