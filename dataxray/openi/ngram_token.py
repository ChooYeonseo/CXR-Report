import re
import glob
import json
import random
import sentencepiece as spm

# Params
word_size_L = [300, 850, 900, 1000, 1200, 1700, 2000, 2770] # Most frequent words
word_size = 1300
total_size = 1460 # total_size = word_size + others
dataset_dir = '/Users/sean/Seans Mac Pro/Programming_Projects/AI/ChestXrayReportGen/dataxray/openi/'

# Reading json file to dictionary
def read_json_data(dir):
    with open(dir + "captions.json", "r") as f:
        data = json.load(f)
    
    return data

# Compute covering ratio
def get_wordfreq_dic(data):
    wordfreq = dict()
    for k, v in data.items():
        for token in re.findall(r'\b[a-zA-Z]+\b', v):
            if token not in wordfreq:
                wordfreq[token] = 1
            else:
                wordfreq[token]+= 1
    
    wordfreq = sorted([(k,v) for k,v in wordfreq.items()], key=lambda x: x[1], reverse=True)
    return wordfreq

# Getting the best word_size value (selecting the hyperparameter)
def coverage(wordfreq, word_size_L):
    result = dict()
    for i in word_size_L:
        total_filter = 0
        num = 0
        for k,v in wordfreq[:i]:
            total_filter += v


        total = 0
        for k,v in wordfreq[:]:
            total += v
            num += 1
        
        result[i] = total_filter / total
    
    print(num)
    print(result)
    
    return None

def make_txt(dir, dic):
    filename = str(dir + 'captions_value.txt')
    with open(filename, "w") as f:
        for _, value in dic.items():
            f.write(value + "\n")
    
    

#######################################################################
####################### Main Work Space ###############################
#######################################################################


data_dict = read_json_data(dataset_dir)
wordfreq = get_wordfreq_dic(data_dict)
coverage(wordfreq, word_size_L)
sub_wordfreq = wordfreq[:word_size]
make_txt(dataset_dir, data_dict)

word_list = [k for k,v in sub_wordfreq]
punc_list = ['`','~','!','@','#','$','%','^','&','*','-','_','+','=','\\','|',':',';','"','\'',',','.','?','/','(',')','{','}','[',']','<','>']

mode_type = 'unigram'
spm.SentencePieceTrainer.train(
    input=dataset_dir + "captions_value.txt",
    model_prefix=dataset_dir + 'NLMCXR_{}_{}'.format(mode_type,total_size), 
    vocab_size=total_size, 
    model_type=mode_type, 
    unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    user_defined_symbols=punc_list + word_list,
)

vocab = spm.SentencePieceProcessor(model_file=dataset_dir + 'NLMCXR_{}_{}.model'.format(mode_type,total_size))
test_data = vocab.encode(random.choice(list(data_dict.values())), out_type=int)
print(vocab.id_to_piece(test_data))
print(sub_wordfreq)