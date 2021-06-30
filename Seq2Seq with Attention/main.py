print('lets do it')
from processing import prepare_dataset
from model import model_construct
from training import train
import torch.nn as nn



class Config():
    dim_embed = 32,
    dim_model = 32,
    batch_size = 16,
    num_layers = 2,
    is_bidirection = True,
    truncated = 5,
    dropout_ratio = 0.3,
    base_path = '/content/drive/MyDrive/04_프로젝트/seq2seq with attention/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    epochs = 30
    clip = 1




def main():
    config = Config()
    (train_iter, dev_iter, test_iter), (ko_vocab, en_vocab) = prepare_dataset(config.base_path)
    model = model_construct(config.dim_model, config.dim_embed, config.batch_size, config.num_layers, config.is_bidirection, config.truncated, config.dropout_ratio, en_vocab, ko_vocab, config.device)
    criterion = criterion = nn.CrossEntropyLoss(ignore_index=ko_vocab["<PAD>"])

    train(config.epochs, config.clip, model, train_iter, dev_iter, test_iter,criterion, config.device)


if __name__ == '__main__':
    main()

