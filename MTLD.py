import string
import argparse
import nltk
import numpy as np
import random
from nltk.collocations import *

# Global trandform for removing punctuation from words
from regex import Regex

remove_punctuation = str.maketrans('', '', string.punctuation)  # return a dict with unicode

# MTLD internal implementation
def mtld_calc(word_array, ttr_threshold):  # ttr_threshold is a score
    current_ttr = 1.0
    token_count = 0
    type_count = 0
    types = set()
    factors = 0.0

    for token in word_array:
        token = token.translate(remove_punctuation).lower()  # trim punctuation, make lowercase
        token_count += 1
        if token not in types:
            type_count += 1
            types.add(token)
        current_ttr = type_count / token_count
        if current_ttr <= ttr_threshold:  # it means a lot of repeat
            factors += 1
            token_count = 0
            type_count = 0
            types = set()
            current_ttr = 1.0

    excess = 1.0 - current_ttr
    excess_val = 1.0 - ttr_threshold
    factors += excess / excess_val
    if factors != 0:
        return len(word_array) / factors
    return -1

# MTLD implementation
def mtld(word_array, ttr_threshold=0.72):
    if isinstance(word_array, str):
        raise ValueError("Input should be a list of strings, rather than a string. Try using string.split()")
    # if len(word_array) < 50:
    #     print(word_array)
    #     raise ValueError("Input word list should be at least 50 in length")
    if len(word_array) < 8:
        raise ValueError("Input word list should be at least 8 in length")
    return (mtld_calc(word_array, ttr_threshold) + mtld_calc(word_array[::-1], ttr_threshold)) / 2



def mtld_calc2(word_array, ttr_threshold):  # ttr_threshold is a score
    current_ttr = 1.0
    token_count = 0
    type_count = 0
    types = set()
    factors = 0.0

    for token in word_array:
        token = token.translate(remove_punctuation).lower()  # trim punctuation, make lowercase
        token_count += 1
        if token not in types:
            type_count += 1
            types.add(token)
        current_ttr = type_count / token_count
        if current_ttr <= ttr_threshold:  # it means a lot of repeat
            factors += 1
            token_count = 0
            type_count = 0
            types = set()
            current_ttr = 1.0

    excess = 1.0 - current_ttr
    excess_val = 1.0 - ttr_threshold
    factors += excess / excess_val
    if factors != 0:
        return len(word_array) / factors
    return -1


# MTLD implementation
def mtld2(word_array, ttr_threshold=0.72):
    if isinstance(word_array, str):
        raise ValueError("Input should be a list of strings, rather than a string. Try using string.split()")
    if len(word_array) < 50:
        print(word_array)
        raise ValueError("Input word list should be at least 50 in length")
    # if len(word_array) < 8:
    #     raise ValueError("Input word list should be at least 8 in length")
    return (mtld_calc2(word_array, ttr_threshold) + mtld_calc2(word_array[::-1], ttr_threshold)) / 2

def data_wash():
    with open('traffic_new','r',encoding='utf-8') as f:
        with open('traffic_new_2','w',encoding='utf-8') as w:
            lines = f.readlines()
            i = 0
            for line in lines:
                line = line.strip('\n')
                lines = line.split(' @@')
                code_prompt = lines[0]
                text = lines[1]
                if i == 0:
                    w.write(str(line) + ' && ')
                    i += 1
                elif i == 7: # batch_size=8
                    w.write(text + '\n')
                    i = 0
                else:
                    w.write(text + ' && ')
                    i += 1

def data_wash2():
    with open('traffic_new_2', 'r', encoding='utf-8') as f:
        with open('traffic_new_5', 'w', encoding='utf-8') as w:
            mean_MTLD = []
            lines = f.readlines()
            i = 0
            print(len(lines))
            for line in lines:
                if len(line.split()) < 50:
                    continue
                MTLD_line = mtld(line.split())
                # mean_MTLD.append(MTLD_line)

                if MTLD_line > 40:
                    mean_MTLD.append(MTLD_line)
                    line = line.strip('\n')
                    lines = line.split(' @@')
                    code_prompt = lines[0]
                    text = lines[1]
                    text_list = text.split(' && ')
                    for text_i in text_list:
                        i += 1
                        w.write(code_prompt+' @@'+text_i+'\n')
            print(i)
            score = np.mean(mean_MTLD)
            print(score)

            #     line = line.strip('\n')
            #     w.write(str(MTLD_line)+'\t'+line+'\n')
            # print(np.mean(mean_MTLD))

def data_wash_score_look():

    with open('traffic_new_score', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sum_score = []
        i=0
        for line in lines:
            line_split = line.split('\t')
            score = line_split[0]
            sum_score.append(float(score))
            if float(score) > 40:
                print(line)
                i += 1
        print(i)
        print(len(lines))
        print(np.mean(sum_score)) # 25

def data_wash3():
    f = open('control-prefixes_1_0.6_data_all', 'r', encoding='utf-8')
    book = open('book', 'w', encoding='utf-8')
    menu = open('menu', 'w', encoding='utf-8')
    traffic = open('traffic', 'w', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        line_split = line.split('\t')
        if line_split[0] == '|menu':
            menu.write(line+'\n')
        elif line_split[0] == '|book':
            book.write(line+'\n')
        elif line_split[0] == '|traffic':
            traffic.write(line+'\n')

def data_wash4(): # shuffle
    f = open('traffic_new', 'r', encoding='utf-8')
    w = open('traffic_new_4', 'w', encoding='utf-8')
    lines = f.readlines()
    i = 0
    temp = []
    temp_sum = []
    line0 = lines[0].split(' @@')
    code_prompt = line0[0]
    for line in lines:
        line = line.strip('\n')
        lines = line.split(' @@')
        if code_prompt == lines[0]: # same prompt
            temp.append(lines[1])
        else: # different prompt
            temp.append(code_prompt) # the last one save the prompt
            temp_sum.append(temp) # [[[text1][text2]...(same prompt)]]
            temp = []
            code_prompt = lines[0]

    for idx, temp_i in enumerate(temp_sum):
        code_prompt = temp_i[-1]
        test = code_prompt + '\t'
        temp_i = temp_i[:-2]
        for i in range(5):
            j = 1
            random.shuffle(temp_i)

            for lines in temp_i:
                line = lines.strip('\n')
                if j % 8 != 0:
                    test += line
                    test += '&&'
                    j += 1
                else:
                    test += line
                    if len(test.split()) < 50:
                        temp_i[j - 8:j] = []
                        j += 1
                        test = code_prompt + '\t'
                        continue
                    MTLD_line = mtld(test.split())
                    if MTLD_line < 25:
                        temp_sum[idx][j-8:j] = []
                        temp_i[j-8:j] = []
                    j += 1
                    test = code_prompt + '\t'
        for line in temp_i:
            if line != []:
                w.write(code_prompt+' @@'+line +'\n')

def data_wash_cut():
    with open('traffic_new_3', 'r', encoding='utf-8') as f:
        with open('traffic_new_6', 'w', encoding='utf-8') as w:
            lines = f.readlines()
            print(len(lines))
            i = 0
            for line in lines: # |traffic	Metal bridge deck @@It's the most expensive bridge deck ever built.
                line = line.strip('\n')
                line_split = line.split(' @@')
                text = line_split[1]
                prompt = line_split[0].split('\t')[1]
                if len(text.split()) <= len(prompt.split())+1:
                    continue
                w.write(line + '\n')
                i += 1
            print(i)

def build_menu_dict():
    with open('traffic', 'r', encoding='utf-8') as f:
        dict = set()
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            dict.add(line)
    return dict

def data_wash_5(menu_dict):
    with open('traffic', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        with open('menu_new_4', 'w', encoding='utf-8') as w:
            for line in lines:
                line_prompt = line.split(' @@')[0].split('\t')[1]
                if line_prompt in menu_dict:
                    w.write(line)

def data_wash_6():
    with open('traffic_new_6', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = set(lines)
        mean_MTLD = []
        with open('traffic_new_7', 'w', encoding='utf-8') as w:
            for line in lines:
                list_char = []
                for char in line:
                    list_char.append(char)
                if len(line) < 50:
                    continue
                MTLD_line = mtld2(list_char)
                if MTLD_line > 10:
                    w.write(line)
                    mean_MTLD.append(MTLD_line)

            print(np.mean(mean_MTLD))






def main():
    # data_wash()
    # data_wash2()
    # data_wash_score_look()
    # data_wash3()
    # data_wash4()
    # data_wash_cut()
    # menu_dict = build_menu_dict()
    # data_wash_5(menu_dict)
    data_wash_6()

if __name__ == '__main__':
    main()



