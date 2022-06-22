# pip install smart-pytorch
from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
import numpy as np
import torch.nn.functional as F
from tasks.glue.dataset import bulid_data, Dataset_BERT, build_iterator
from my_bertmodel import Config, PromptBERT, get_emb, PromptBERT_new, PromptRoberta, BERT_new, bert_gan
from dataset2 import bulid_dataset, build_iterator
from transformers import AdamW
import torch
from utils import get_time_dif
import time
from transformers import logging, AutoTokenizer
import torch.nn as nn
logging.set_verbosity_error() # cancel warning
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(): # 八分类
    emb = get_emb()
    config = Config()
    model = PromptBERT_new()
    # model = PromptRoberta()
    # model = BERT_new()
    model = model.to(config.device)

    model.to(config.device)


    # train_data = bulid_data(config.tokenizer, path='/common-data/new_build/xiaoying.huang/emotion-main/my_model1/tasks/glue/sst2/train.jsonl')
    # train_iter = build_iterator(train_data, config)
    # dev_data = bulid_data(config.tokenizer, path='/common-data/new_build/xiaoying.huang/emotion-main/my_model1/tasks/glue/sst2/dev.jsonl')
    # dev_iter = build_iterator(dev_data, config)
    # test_data = bulid_data(config.tokenizer, path='/common-data/new_build/xiaoying.huang/emotion-main/my_model1/tasks/glue/sst2/test.jsonl')
    # test_iter = build_iterator(test_data, config)

    train_data = bulid_dataset(config, type='train')
    train_iter = build_iterator(train_data, config)
    dev_data = bulid_dataset(config, type='dev')
    dev_iter = build_iterator(dev_data, config)
    test_data = bulid_dataset(config, type='test')
    test_iter = build_iterator(test_data, config)

    dev_best_loss = float('inf')
    dev_best_acc = float(0)
    start_time = time.time()
    total_batch = 0  # 记录进行到多少batch
    last_improve = 0  # 记录上次验证集loss下降的batch数
    total_loss = 0
    total_adv_loss = 0
    flag = False  # 记录是否很久没有效果提升
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)


    # smart_adv = SmartPerturbation(model)
    smart_loss_fn = SMARTLoss(eval_fn=model, loss_fn=kl_loss, loss_last_fn=sym_kl_loss)
    model.train()

    for epoch in range(config.num_epochs):
        logging.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, batch in enumerate(train_iter): # token_ids, mask, label
            (input_ids, mask), label = batch
            new_emb, new_mask = emb.forward(input_ids, mask)
            outputs = model(input_emb=new_emb, attention_mask=new_mask, label=label)

            # outputs = model(input_ids=input_ids, attention_mask=mask, label=label)

            model.zero_grad()
            loss = outputs.loss
            total_loss += loss

            # loss_adv = smart_adv.forward(input_emb=new_emb, attention_mask=new_mask, label=label, logits=outputs.logits)
            # loss = loss - 0.001 * loss_adv

            loss += 0.02 * smart_loss_fn(new_emb, new_mask, label, outputs.logits)
            logits = outputs.logits.argmax(dim=1)

            total_adv_loss += loss
            loss.backward()
            optimizer.step()

            predicted_token_id = logits.cpu().numpy()
            label = label.cpu().numpy()
            predict_all = np.append(predict_all, predicted_token_id)
            labels_all = np.append(labels_all, label)

            if i % 500 == 499:
                acc = 0
                acc = ((predict_all == labels_all).sum()).item()
                # for i in range(len(labels_all)):
                #     acc += 1 if all(labels_all[i] == predict_all[i]) else 0
                train_acc = acc / len(labels_all)

                predict_all = np.array([], dtype=int)
                labels_all = np.array([], dtype=int) # 重置
                dev_acc, dev_loss = evaluate(config, emb, model, dev_iter)
                # if dev_loss < dev_best_loss:
                #     dev_best_loss = dev_loss
                #     torch.save(model.state_dict(), config.save_path)
                #     improve = '*'
                #     last_improve = total_batch
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter:{0:>4}, Train Loss:{1:>5.2}, Train Acc:{2:>6.2%}, Val Loss:{3:>5.2}, Best Val acc:{4:>6.2%}, ' \
                      'Val Acc: {5:>6.2%}, Time: {6} {7}, adv_loss: {8:>5.2}'

                logging.info(msg.format(total_batch, total_loss/500, train_acc, dev_loss, dev_best_acc, dev_acc, time_dif, improve, total_adv_loss/500))
                model.train()
                total_loss = 0
                total_adv_loss = 0
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                logging.info("No optimization for a long time, auto-stopping...")
                test(config, emb, model, test_iter)
                flag = True
                break
        if flag:
            break
        test(config, emb, model, test_iter)

def train2(): # 二分类
    # emb = get_emb()
    config = Config()
    # model = PromptBERT_new()
    # model = PromptRoberta()
    # model = BERT_new()
    model = bert_gan()
    model = model.to(config.device)

    model.to(config.device)


    train_data = bulid_data(config.tokenizer, path='/common-data/new_build/xiaoying.huang/emotion-main/my_model1/tasks/glue/sst2/train.jsonl')
    train_iter = build_iterator(train_data, config)
    dev_data = bulid_data(config.tokenizer, path='/common-data/new_build/xiaoying.huang/emotion-main/my_model1/tasks/glue/sst2/dev.jsonl')
    dev_iter = build_iterator(dev_data, config)
    test_data = bulid_data(config.tokenizer, path='/common-data/new_build/xiaoying.huang/emotion-main/my_model1/tasks/glue/sst2/test.jsonl')
    test_iter = build_iterator(test_data, config)

    # train_data = bulid_dataset(config, type='train')
    # train_iter = build_iterator(train_data, config)
    # dev_data = bulid_dataset(config, type='dev')
    # dev_iter = build_iterator(dev_data, config)
    # test_data = bulid_dataset(config, type='test')
    # test_iter = build_iterator(test_data, config)

    dev_best_loss = float('inf')
    dev_best_acc = float(0)
    start_time = time.time()
    total_batch = 0  # 记录进行到多少batch
    last_improve = 0  # 记录上次验证集loss下降的batch数
    total_loss = 0
    total_adv_loss = 0
    flag = False  # 记录是否很久没有效果提升
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)


    # smart_adv = SmartPerturbation(model)
    smart_loss_fn = SMARTLoss(eval_fn=model, loss_fn=kl_loss, loss_last_fn=sym_kl_loss)
    model.train()

    for epoch in range(config.num_epochs):
        logging.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, batch in enumerate(train_iter): # token_ids, mask, label
            (input_ids, mask), label = batch
            new_emb, new_mask = model.get_emb(input_ids, mask)
            outputs = model(input_emb=new_emb, attention_mask=new_mask, label=label)

            # outputs = model(input_ids=input_ids, attention_mask=mask, label=label)

            model.zero_grad()
            loss = outputs.loss
            total_loss += loss

            # loss_adv = smart_adv.forward(input_emb=new_emb, attention_mask=new_mask, label=label, logits=outputs.logits)
            # loss = loss - 0.001 * loss_adv

            loss += 0.02 * smart_loss_fn(new_emb, new_mask, label, outputs.logits)
            logits = outputs.logits.argmax(dim=1)

            total_adv_loss += loss
            loss.backward()
            optimizer.step()

            predicted_token_id = logits.cpu().numpy()
            label = label.cpu().numpy()
            predict_all = np.append(predict_all, predicted_token_id)
            labels_all = np.append(labels_all, label)

            if i % 500 == 499:
                acc = 0
                acc = ((predict_all == labels_all).sum()).item()
                # for i in range(len(labels_all)):
                #     acc += 1 if all(labels_all[i] == predict_all[i]) else 0
                train_acc = acc / len(labels_all)

                predict_all = np.array([], dtype=int)
                labels_all = np.array([], dtype=int) # 重置
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # if dev_loss < dev_best_loss:
                #     dev_best_loss = dev_loss
                #     torch.save(model.state_dict(), config.save_path)
                #     improve = '*'
                #     last_improve = total_batch
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter:{0:>4}, Train Loss:{1:>5.2}, Train Acc:{2:>6.2%}, Val Loss:{3:>5.2}, Best Val acc:{4:>6.2%}, ' \
                      'Val Acc: {5:>6.2%}, Time: {6} {7}, adv_loss: {8:>5.2}'

                logging.info(msg.format(total_batch, total_loss/500, train_acc, dev_loss, dev_best_acc, dev_acc, time_dif, improve, total_adv_loss/500))
                model.train()
                total_loss = 0
                total_adv_loss = 0
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                logging.info("No optimization for a long time, auto-stopping...")
                test(config, model, test_iter)
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
    logging.info(msg.format(test_loss, test_acc))

    time_dif = get_time_dif(start_time)
    logging.info("Time usage:", time_dif)
    with open('log.txt', 'a', encoding='utf-8') as f:
        f.write(f'Model_name: {config.model_name},   Test Loss: {test_loss},  Test Acc: {test_acc} \n')

def evaluate(config, model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_iter:
            (input_ids, mask), label = batch
            # new_emb, new_mask = emb.forward(input_ids, mask)
            new_emb, new_mask = model.get_emb(input_ids, mask)
            outputs = model(input_emb=new_emb, attention_mask=new_mask, label=label)

            loss = outputs.loss
            logits = outputs.logits.argmax(dim=1)
            loss_total += loss
            predicted_token_id = logits.cpu().numpy()
            label = label.cpu().numpy()
            predict_all = np.append(predict_all, predicted_token_id)
            labels_all = np.append(labels_all, label)

    # acc = 0
    # for i in range(len(labels_all)):
    #      print(labels_all[i], predict_all[i])
    #      if labels_all[i] == predict_all[i]:
    #          acc += 1

    acc = ((predict_all == labels_all).sum()).item()
    acc = acc / len(labels_all)


    return acc, loss_total / len(data_iter)


def main():
    train2()
    # train()


if __name__ == '__main__':
    main()

