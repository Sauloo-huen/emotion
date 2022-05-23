from torch.utils.data import Dataset
import torch

train_path = 'data/Train1.txt'
test_path = 'data/Test1.txt'

def bulid_dataset(config, type):
    def build_dataset_(path, pad_size=60):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                label, text = line.split('\t')
                labels = []
                mask_token_index = []
                for str in label:
                    labels.append(str)
                if len(text) > pad_size-10:
                    text = text[0:pad_size-10]
                token = '[CLS] ' + text + ' [SEP] [SEP] ' + '这真让人[MASK][MASK]。' + ' [SEP]'
                token = config.tokenizer.tokenize(token)
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                mask_token_ids= token_ids.index(config.tokenizer.mask_token_id)
                mask_token_index.append(mask_token_ids)
                mask_token_index.append(mask_token_ids+1)
                # mask_token_index = (token_ids == config.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)
                label = config.tokenizer.convert_tokens_to_ids(labels)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                data.append((token_ids, label, seq_len, mask, mask_token_index))
        print(f'{len(data)}data has load!')
        return data

    if type == 'train':
        path = r'D:\Python_projection\Bert-Chinese-Prompt-Mask-main\data\Train1.txt'
    elif type == 'dev':
        path = r'D:\Python_projection\Bert-Chinese-Prompt-Mask-main\data\Dev.txt'
    else:
        path = r'D:\Python_projection\Bert-Chinese-Prompt-Mask-main\data\Test2.txt'
    return build_dataset_(path)

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
        # token_ids, label, seq_len, mask
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device) # _[0]即每个batch里面的Token_ids
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        mask_token_index = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y, mask_token_index

    def __next__(self):
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
        return self

    def __len__(self):
        if self.residue: # 不能被整除时
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = Dataset_BERT(dataset, config.batch_size, config.device)
    return iter

def main():
    from my_bertmodel import Config
    config = Config()
    bulid_dataset(config, 'test')

if __name__ == '__main__':
    main()



