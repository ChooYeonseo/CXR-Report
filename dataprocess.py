import os
import json
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from random import shuffle
import sentencepiece as spm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class NLMCXR(data.Dataset): # Open-I Dataset
    def __init__(self, directory, input_size=(256,256), random_transform=True,
                view_pos=['AP', 'PA', 'LATERAL'], max_views=2, sources=['image','history'], targets=['label'], 
                max_len=1000, vocab_file='NLMCXR_unigram_1460.model'):
        
        self.source_sections = ['INDICATION', 'COMPARISON']
        self.target_sections = ['FINDINGS']
        self.vocab = spm.SentencePieceProcessor(model_file=directory + vocab_file)
        self.vocab_file = vocab_file # Save it for subsets

        self.sources = sources # Choose which section as input
        self.targets = targets # Choose which section as output
        self.max_views = max_views
        self.view_pos = view_pos
        self.max_len = max_len

        self.dir = directory
        self.input_size = input_size
        self.random_transform = random_transform
        self.__input_data(binary_mode=True)
        
        if random_transform:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.1,0.1,0.1), 
                    transforms.RandomRotation(15, expand=True)]),
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        sources, targets = [], []
        tmp_rep = self.captions[self.file_report[file_name]['image'][0] + '.png']
        
        # ------ Multiview Images ------
        if 'image' in self.sources:
            imgs, vpos = [], []
            images = self.file_report[file_name]['image']

            # Randomly select V images from each folder 
            new_orders = np.random.permutation(len(images))
            img_files = np.array(images)[new_orders].tolist()

            for i in range(min(self.max_views,len(img_files))):
                img_file = self.dir + 'images/' + img_files[i] + '.png'
                img = Image.open(img_file).convert('RGB')
                imgs.append(self.transform(img).unsqueeze(0)) # (1,C,W,H)
                vpos.append(1) # We do not know what view position of the image is, so just let it be 1
                
            # If the number of images is smaller than V, pad the tensor with dummy images
            cur_len = len(vpos)
            for i in range(cur_len, self.max_views):
                imgs.append(torch.zeros_like(imgs[0]))
                vpos.append(-1) # Empty mask
            
            imgs = torch.cat(imgs, dim=0) # (V,C,W,H)
            vpos = np.array(vpos, dtype=np.int64) # (V)

        # ------ Additional Information ------
        info = self.file_report[file_name]['report']
        
        source_info = []
        for section, content in info.items():
            if section in self.source_sections:
                source_info.append(content)
        source_info = ' '.join(source_info)
        
        encoded_source_info = [self.vocab.bos_id()] + self.vocab.encode(source_info) + [self.vocab.eos_id()]
        source_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        source_info[:min(len(encoded_source_info), self.max_len)] = encoded_source_info[:min(len(encoded_source_info), self.max_len)]

        target_info = []
        for section, content in info.items():
            if section in self.target_sections:
                target_info.append(content)
        # target_info = ' '.join(target_info)
        target_info = tmp_rep # This load the document from our previous AAAI paper (preprocessed documents)
        
        np_labels = np.zeros(len(self.top_np), dtype=float)
        for i in range(len(self.top_np)):
            if self.top_np[i] in target_info:
                np_labels[i] = 1
        
        encoded_target_info = [self.vocab.bos_id()] + self.vocab.encode(target_info) + [self.vocab.eos_id()]
        target_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        target_info[:min(len(encoded_target_info), self.max_len)] = encoded_target_info[:min(len(encoded_target_info), self.max_len)]

        for i in range(len(self.sources)):
            if self.sources[i] == 'image':
                sources.append((imgs,vpos))
            if self.sources[i] == 'history':
                sources.append(source_info)
            if self.sources[i] == 'label':
                sources.append(np.concatenate([np.array(self.file_labels[file_name]), np_labels]))
            if self.sources[i] == 'caption':
                sources.append(target_info)
            if self.sources[i] == 'caption_length':
                sources.append(min(len(encoded_target_info), self.max_len))
                
        for i in range(len(self.targets)):
            if self.targets[i] == 'label':
                targets.append(np.concatenate([np.array(self.file_labels[file_name]), np_labels]))
            if self.targets[i] == 'caption':
                targets.append(target_info)
            if self.targets[i] == 'caption_length':
                targets.append(min(len(encoded_target_info), self.max_len))
                
        return sources if len(sources) > 1 else sources[0], targets if len(targets) > 1 else targets[0]

    def __get_nounphrase(self, top_k=100, file_name='nounphrase_extnum.json'):
        count_np = json.load(open(self.dir + file_name, 'r'))
        sorted_count_np = sorted([(k,v) for k,v in count_np.items()], key=lambda x: x[1], reverse=True)
        top_nounphrases = [k for k,v in sorted_count_np][:top_k]
        return top_nounphrases

    def __input_data(self, binary_mode=True):
        self.__input_caption()
        self.__input_report()
        self.__input_label()
        self.__filter_inputs()
        self.top_np = self.__get_nounphrase()
        
    def __input_label(self):
        with open(self.dir + 'file2label.json') as f:
            labels = json.load(f)
        self.file_labels = labels
        
    def __input_caption(self):
        with open(self.dir + 'captions.json') as f:
            captions = json.load(f)
        self.captions = captions
        
    def __input_report(self):
        with open(self.dir + 'Full_openi_data.json') as f:
            reports = json.load(f)
        self.file_list = [k for k in reports.keys()]
        self.file_report = reports

    def __filter_inputs(self):
        filtered_file_report = {}
        for k, v in self.file_report.items():
            if (len(v['image']) > 0) and (('FINDINGS' in v['report']) and (v['report']['FINDINGS'] != '')): # or (('IMPRESSION' in v['report']) and (v['report']['IMPRESSION'] != ''))):
                filtered_file_report[k] = v
        self.file_report = filtered_file_report
        self.file_list = [k for k in self.file_report.keys()]

    def get_subsets(self, train_size=0.7, val_size=0.1, test_size=0.2, seed=0):
        np.random.seed(seed)
        indices = np.random.permutation(len(self.file_list))
        train_pvt = int(train_size * len(self.file_list))
        val_pvt = int((train_size + val_size) * len(self.file_list))
        train_indices = indices[:train_pvt]
        val_indices = indices[train_pvt:val_pvt]
        test_indices = indices[val_pvt:]

        master_file_list = np.array(self.file_list)

        train_dataset = NLMCXR(self.dir, self.input_size, self.random_transform, 
                              self.view_pos, self.max_views, self.sources, self.targets, self.max_len, self.vocab_file)
        train_dataset.file_list = master_file_list[train_indices].tolist()

        # Consider change random_transform to False for validation
        val_dataset = NLMCXR(self.dir, self.input_size, False, 
                            self.view_pos, self.max_views, self.sources, self.targets, self.max_len, self.vocab_file)
        val_dataset.file_list = master_file_list[val_indices].tolist()

        # Consider change random_transform to False for testing
        test_dataset = NLMCXR(self.dir, self.input_size, False, 
                             self.view_pos, self.max_views, self.sources, self.targets, self.max_len, self.vocab_file)
        test_dataset.file_list = master_file_list[test_indices].tolist()

        return train_dataset, val_dataset, test_dataset
    
class TextDataset(data.Dataset):
    def __init__(self, text_file, label_file, sources=['caption'], targets=['label'],
                 vocab_file='/Users/sean/Seans Mac Pro/Programming_Projects/AI/ChestXrayReportGen/dataxray/openi/NLMCXR_unigram_1460.model', max_len=1460):
        self.text_file = text_file
        self.label_file = label_file
        self.vocab = spm.SentencePieceProcessor(model_file=vocab_file)
        self.sources = sources # Choose which section as input
        self.targets = targets # Choose which section as output
        self.max_len = max_len
        self.__input_data()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        encoded_text = [self.vocab.bos_id()] + self.vocab.encode(self.lines[idx].strip()) + [self.vocab.eos_id()]
        text = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        text[:min(len(encoded_text), self.max_len)] = encoded_text[:min(len(encoded_text), self.max_len)]
        
        sources = []
        for i in range(len(self.sources)):
            if self.sources[i] == 'label':
                sources.append(self.labels[idx])
            if self.sources[i] == 'caption':
                sources.append(text)
            if self.sources[i] == 'caption_length':
                sources.append(min(len(encoded_text), self.max_len))
        
        targets = []
        for i in range(len(self.targets)):
            if self.targets[i] == 'label':
                targets.append(self.labels[idx])
            if self.targets[i] == 'caption':
                targets.append(text)
            if self.targets[i] == 'caption_length':
                targets.append(min(len(encoded_text), self.max_len))
                
        return sources if len(sources) > 1 else sources[0], targets if len(targets) > 1 else targets[0]
    
    def __input_data(self):
        data_file = open(self.text_file, 'r') 
        self.lines = data_file.readlines()
        self.labels = np.loadtxt(self.label_file, dtype='float')