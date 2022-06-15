import ahocorasick
import json
from tqdm import tqdm
file_list_path = ['/common-data/new_build/xiaoying.huang/datasets/english/news/news.2007.en.shuffled',
                  '/common-data/new_build/xiaoying.huang/datasets/english/news/news.2008.en.shuffled',
                  '/common-data/new_build/xiaoying.huang/datasets/english/news/news.2009.en.shuffled',
                  '/common-data/new_build/xiaoying.huang/datasets/english/news/news.2010.en.shuffled',
                  '/common-data/new_build/xiaoying.huang/datasets/english/news/news.2011.en.shuffled',
                  '/common-data/new_build/xiaoying.huang/datasets/english/news/news.2012.en.shuffled',
                  '/common-data/new_build/xiaoying.huang/datasets/english/news/news.2013.en.shuffled',
                  '/common-data/new_build/xiaoying.huang/datasets/english/news/news.2014.en.shuffled',
                  '/common-data/new_build/xiaoying.huang/datasets/english/news/news.2015.en.shuffled',
                  '/common-data/new_build/xiaoying.huang/datasets/english/news/news.2016.en.shuffled',
                  '/common-data/new_build/xiaoying.huang/datasets/english/news/news.2017.en.shuffled']

def split_sentence(line):
    split_lines = []
    while 1:
        line = line.replace('\n', '')
        temp = line[:71] # until index 70
        line = line[71:]
        split_line = line.split('.', 1)
        # print(line)
        # print('split_line', split_line)
        if split_line[0] == '\n' or split_line[0] == '':
            break
        if split_line[0] == line:
            split_lines.append(split_line)
            break
        temp = temp + split_line[0] + '.'
        split_lines.append(temp)
        line = split_line[1]
        # print(temp)
        # print(line)
    return split_lines

def build_actree(wordlist):
    actree = ahocorasick.Automaton()
    for index, word in enumerate(wordlist):
        actree.add_word(word, (index, word))
    actree.make_automaton()
    return actree

# test = ['abcdefg', 'abcdef', 'abcde','abcd','abc','ab','a','abdcef','cde']

A = ahocorasick.Automaton()
search_file = '/common-data/new_build/xiaoying.huang/datasets/english/news/news.2007.en.shuffled'
# search_keywords_file = 'Dish_wash.txt'
test = []
with open('Dish2.txt', 'r', encoding='utf-8') as search_keywords_file:
    lines = search_keywords_file.readlines()
    lines = set(lines)
    for line in tqdm(lines, total=len(lines)):
        line = line.replace('\n', '')
        test.append(' '+line+' ') # append ' ' to avoid match wrong word
search_keywords_file.close()
print(len(test))
j = 0
temp = []
actree_test = build_actree(test) # build sentence

################################### save ############################

for search_file in file_list_path:
    with open(search_file, 'r', encoding='utf-8') as search_file:
        lines = search_file.readlines()
        with open('menu_all.txt', 'a', encoding='utf-8') as a:
            for line in tqdm(lines, total=len(lines)):
                line.replace('\n', '')
                line.replace('\t', '')
                line = ' '+line+' '
                for val in actree_test.iter(line): # return (search_end_index, (key_index, key_value))
                    # temp.append([j, val]) # j: idx of sentence, i: (search_end_index, (key_index, key_value))
                    # j += 1
                    (_, (key_id, key)) = val
                    length = len(line)
                    line.lstrip(' ')
                    line.rsplit(' ')
                    if length <= 70:
                        line.lstrip('\n')
                        line.rstrip('\n')
                        a.write('|' + key + '\t' + line)
                    elif length > 70:
                        split_lines = split_sentence(line)
                        for split_line in split_lines:
                            if key in split_line:
                                if len(split_line) > 150:
                                    continue
                                else:
                                    split_line = str(split_line)
                                    a.write('|' + key + '\t' + split_line + '\n')
        a.close()
# print(row)
# print(temp)
    search_file.close()
