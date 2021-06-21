import torchtext
import tarfile
import os
import torch
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
import io
mecab = """
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
%cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh
"""
import subprocess
process = subprocess.Popen(mecab.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
from konlpy.tag import Mecab


def prepare_dataset(base_path):
    path = base_path + 'dataset/'
    os.chdir(path)
    save_name = 'unzipped/'
    file_list = os.listdir(path)
    for file in file_list:
        tar = tarfile.open(file)
        tar.extractall('./' + save_name + file.split(".")[0])
        tar.close()

    os.chdir(base_path)

    unzipped_path = "./dataset/unzipped/korean-english-park/korean-english-park."
    train_path = ('train.en', 'train.ko')
    dev_path = ('dev.en', 'dev.ko')
    test_path = ('test.en', 'test.ko')

    


    en_tokenizer = get_tokenizer('spacy')
    ko_tokenizer = Mecab()    
    
    # 출처 : https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html

    ko_vocab = build_vocab(unzipped_path + train_path[1], ko_tokenizer)
    en_vocab = build_vocab(unzipped_path + train_path[0], en_tokenizer) 

    train_data= data_preprocess([unzipped_path + path for path in train_path])
    dev_data = data_preprocess([unzipped_path + path for path in dev_path])
    test_data = data_preprocess([unzipped_path + path for path in test_path])

    PAD_IDX = ko_vocab['<PAD>']
    BOS_IDX = ko_vocab['<BOS>']
    EOS_IDX = ko_vocab['<EOS>']

    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
    dev_iter = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
    test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

    return (train_iter, dev_iter, test_iter), (ko_vocab, en_vocab)






def build_vocab(train_path, tokenizer):
    counter = Counter()
    with open(train_path, encoding = 'UTF-8', newline = '\n') as f:
        for string_ in f:
            if 'ko' in train_path[-10:]:
                counter.update(tokenizer.morphs(string_))
            else:   
                counter.update(tokenizer(string_))
        return Vocab(counter, min_freq = 3, specials = ('<unk>', '<BOS>', '<EOS>', "<PAD>"))


def data_preprocess(file_paths):
  raw_ko_iter = iter(io.open(file_paths[1], encoding = 'UTF-8', newline = '\n'))
  raw_en_iter = iter(io.open(file_paths[0], encoding = 'UTF-8', newline = '\n'))
  data = []
  for raw_ko, raw_en in zip(raw_ko_iter, raw_en_iter):
    ko_tensor = torch.tensor([ko_vocab[token] for token in ko_tokenizer.morphs(raw_ko)], dtype = torch.long)
    en_tensor = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)], dtype = torch.long)
    data.append((ko_tensor, en_tensor))

  return data


def generate_batch(data_batch):
  ko_batch, en_batch = [], []
  for (ko_item, en_item) in data_batch:
    ko_batch.append(torch.cat([torch.tensor([BOS_IDX]), ko_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
  ko_batch = pad_sequence(ko_batch, padding_value=PAD_IDX, batch_first = True)
  en_batch = pad_sequence(en_batch, padding_value=PAD_IDX, batch_first = True)
  return ko_batch, en_batch

        
