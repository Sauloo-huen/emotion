import pandas as pd
import tqdm
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer



def load_config_for_control_prefixes(path_data): # for txt
    """
    Dict of Control-Codes in Dataset.
    Need for Training Control-Prefixes Model.
    e.g.
    {
        ' positive ': 1,
        ' negative ': 2
    }
    or
    {
        ' economy ': 1,
        ' society ': 2,
        ' sports ': 3,
        ...
    }
    """
    dict_codes={}
    with open(path_data, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            code = line.split('\t')[0]
            for c in code.split('|')[1:]:
                if c not in dict_codes: dict_codes[c] = len(dict_codes) + 1
    f.close()

    return dict_codes

class DatasetControlPrefixes(Dataset): # txt dataset
    """
    """

    def __init__(self, path_data, tokenizer):
        self.data = []
        self.label = []
        # Control-Code (Class)
        self.control = []

        print('Processing Data..')

        # Load Data
        with open(path_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # '|' + 'traffic' + '\t' + key + '@@' + line + '\n'
                code = line.split('\t')[0]
                text1 = line.split('\t')[1]
                # text = line.split('\t')[1].split(' @@')
                # text1 = text[1]
                # prompt = text[0].lstrip() # '#toy#'

                # Debug
                if type(code) != str or type(text1) != str: continue

                # data = tokenizer.encode(code + '\t' + prompt + ' @@' + tokenizer.bos_token + text1 + tokenizer.eos_token)
                data = tokenizer.encode(code + tokenizer.bos_token + text1 + tokenizer.eos_token)
                # print(data)
                self.data.append(data)

                # Label: -100 -100 ... -100 Text <EOS>
                label = tokenizer.encode(code + tokenizer.bos_token + text1 + tokenizer.eos_token)
                sep1 = data.index(tokenizer.bos_token_id) + 1 # text + tokenizer.eos_token
                # sep2 = data.index(prompt)
                # sep3 = data.index(tokenizer.sep_token_id) + 1

                # Masking
                label[:sep1] = [-100] * sep1 # after mask, only text + tokenizer.eos_token
                # label[sep2:sep3] = [-100] * sep1
                self.label.append(label)

                # Control-Code: '| economy | sports ':str -> [' economy ', ' sports ']:list
                self.control.append(code.split('|')[1:])

            print(len(self.data), 'Data Processed!\n')
        f.close()

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.control[idx]

    def __len__(self):
        return len(self.data)

def collate_fn_control_prefixes(pad_token_id):
    """
    """
    def collate_fn(batch):
        max_len=0
        max_len_control=0
        for data, _, control in batch:
            if len(data)>max_len: max_len=len(data)
            if len(control)>max_len_control: max_len_control=len(control)
                
        datas=[]
        labels=[]
        controls=[]
        for data, label, control in batch:
            data.extend([pad_token_id]*(max_len-len(data)))
            datas.append(data)
            
            label.extend([pad_token_id]*(max_len-len(label)))
            labels.append(label)

            control.extend(['pad']*(max_len_control-len(control)))
            controls.append(control)
            
        return torch.tensor(datas), torch.tensor(labels), controls

    return collate_fn


def main():
    path_data = '/common-data/new_build/xiaoying.huang/123/dataset/parapat2'
    tokenizer = GPT2Tokenizer.from_pretrained('/common-data/new_build/xiaoying.huang/123/Wenzhong2.0-GPT2-3.5B')
    dataset = DatasetControlPrefixes(path_data, tokenizer)
    print(dataset)

if __name__ == '__main__':
    main()


