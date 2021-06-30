import torch
import torch.nn as nn
import numpy as np
import copy
from torch.nn import Dropout



def model_construct(dim_model, dim_embed, batch_size, num_layers, is_bidirection, truncated, dropout_ratio, encoder_vocab, decoder_vocab, device):
    model = Model(Encoder, Decoder, dim_model, dim_embed, batch_size, num_layers, gru, gru_cell, is_bidirection, truncated, dropout_ratio, encoder_vocab, decoder_vocab, device)
    return model






class Model(nn.Module):
  def __init__(self, Encoder, Decoder, dim_model, dim_embed, batch_size, num_layers, gru, gru_cell, is_bidirection, truncated, dropout_ratio, encoder_vocab, decoder_vocab, device):
    super().__init__()
    self.device = device

    self.encoder_vocab = encoder_vocab
    self.decoder_vocab = decoder_vocab
    self.encoder_vocab_size = len(self.encoder_vocab)
    self.decoder_vocab_size = len(self.decoder_vocab) 

    encoder_partial = Encoder(dim_model, dim_embed, batch_size, 3, gru, gru_cell, truncated, bidirection = is_bidirection)
    encoder_embedding = nn.Embedding(self.encoder_vocab_size, dim_embed)
    encoder_dropout = Dropout(dropout_ratio)
    encoder = nn.Sequential(encoder_embedding, encoder_dropout, encoder_partial)

    decoder = Decoder(dim_model, dim_embed, batch_size, encoder_bidirectional = is_bidirection)
    linear_input_dim = 2*dim_model if is_bidirection else dim_model
    decoder_linear = nn.Linear(linear_input_dim, self.decoder_vocab_size)
    
    self.layers = nn.ModuleDict({"encoder" : encoder, "decoder" : decoder, "dense" : decoder_linear})


    self.__sos_token_index = torch.tensor(self.decoder_vocab.stoi["<BOS>"], device = self.device)
    self.__eos_token_index = torch.tensor(self.decoder_vocab.stoi["<EOS>"], device = self.device)
    self.__pad_token_index = torch.tensor(self.decoder_vocab.stoi['<PAD>'], device = self.device)
    self.batch_size = batch_size
    self.decoder_embedding = nn.Sequential(nn.Embedding(num_embeddings = self.decoder_vocab_size, embedding_dim = dim_embed, padding_idx = self.__pad_token_index))   
    
    

  def forward(self, input_tokens, gold_tokens, teach_forcing_ratio = 0.7): # input_tokens : 번역할 문장 (batch_size, max_seq) gold_token : 실제 정답 토큰 <sos> 부착 상태 (batch_size, max_seq_decoder)
    gold_embed = self.decoder_embedding(gold_tokens) # (batch_size, max_seq, dim_embed)
    encoder_max_seq = input_tokens.size()[-1]
    encoding_matrix, hidden_state = self.layers.encoder.forward(input_tokens)

    output_embed = self.decoder_embedding(self.__sos_token_index).squeeze(0).expand(self.batch_size, dim_embed) # (batch_size, dim_embed)
    max_seq_decoder = gold_tokens.size()[-1]
    teach_forcing_prob = torch.rand(max_seq_decoder)

    sentence_made = torch.zeros((batch_size, max_seq_decoder, self.decoder_vocab_size))

    for seq in range(max_seq_decoder):
      hidden_state = self.layers.decoder.forward(output_embed, hidden_state, encoding_matrix)  #(batch_size, 2*dim_model)
      
      output_token_vect = self.layers.dense(hidden_state)
      sentence_made[:, seq, :] = output_token_vect # 손실 함수를 위해 출력할 값 저장. 이 값은 softmax 통과하기 전 (batch_size, decoder_vocab_size)
      output_token = output_token_vect.argmax(dim = 1) # 어차피 소프트맥스 하나 안하나 이 시점의 가장 큰 값이 output token
      output_embed = self.decoder_embedding(output_token)
      output_embed = gold_embed[:, seq, :] if (teach_forcing_prob[seq] < teach_forcing_ratio) else output_embed # teacher forcing을 사용할지 정해짐

    return sentence_made 


class gru_cell(nn.Module):
  def __init__(self, dim_model, dim_embed, batch_size):
    super().__init__()
    self.W_z = nn.Linear(dim_embed, dim_model)
    self.W_r = nn.Linear(dim_embed, dim_model)
    self.W_ = nn.Linear(dim_embed, dim_model)
    self.U_z = nn.Linear(dim_model, dim_model)
    self.U_r = nn.Linear(dim_model, dim_model)
    self.U_ = nn.Linear(dim_model, dim_model)
    self.h = torch.zeros(dim_model, device = device).repeat(batch_size, 1)
  def forward(self, x, h_old = None): # x : 모든 배치의 한 시점의 단어 임베딩 벡터 (batch_size, dim_embed)
    if h_old is None:
      h_old = self.h
    r = torch.sigmoid(self.W_r(x) + self.U_r(h_old))
    z = torch.sigmoid(self.W_z(x) + self.U_z(h_old))
    h_new = torch.tanh(self.W_(x) + self.U_(r*h_old))
    h = (1 - z)*h_old + z*h_new
    return h


class gru(nn.Module):
  def __init__(self, dim_model, dim_embed, batch_size, gru_cell, truncated):
    super().__init__()
    self.layer = nn.Sequential(gru_cell(dim_model, dim_embed, batch_size))
    self.truncated = truncated
    # self.bptt_truncated = bptt_truncated # bptt를 몇 시점마다 짜를지

  def forward(self, x, max_seq, return_states = True, backward = False):
    h = None
    states = []
    input = x.permute(1, 0, 2) # (time_step, batch_size, dim_model) 로 바꿈.
    for time_step in range(max_seq):
      if backward: # 역방향 레이어면, 맨 뒤의 값부터 입력
        time_step = max_seq - time_step - 1
      h = self.layer[0].forward(input[time_step], h)
      if ((max_seq - self.truncated) % self.truncated == 0): # truncated BPTT, truncated 마다 h의 backprop을 끊음으로써 구현
        h.detach()
      if backward: # 역방향 레이어면 출력 순서도 역방향이므로 다시 돌려서 저장
        states.insert(0, h)
      else: 
        states.append(h)
          
    states = torch.stack(states).permute(1, 0, 2)
    if return_states:
      return states
    else: 
      return self.h


class Encoder(nn.Module):
  def __init__(self, dim_model, dim_embed, batch_size, num_layers, gru, gru_cell, truncated, bidirection = False):
    super().__init__()
    self.direction = 2 if bidirection else 1
    rest_layer_input = 2*dim_model if bidirection else dim_model
    first_forward_floor = gru(dim_model, dim_embed, batch_size, gru_cell, truncated)
    forward_layers = [gru(dim_model, rest_layer_input, batch_size, gru_cell, truncated) for layer in range(num_layers - 1)]
    forward_layers.insert(0, first_forward_floor)
    self.forward_layers = nn.ModuleList(forward_layers)
    if self.direction == 2:
      first_backward_floor = gru(dim_model, dim_embed, batch_size, gru_cell, truncated)
      backward_layers = [gru(dim_model, rest_layer_input, batch_size, gru_cell, truncated) for layer in range(num_layers - 1)]
      backward_layers.insert(0, first_forward_floor)
      self.backward_layers = nn.ModuleList(backward_layers)
    self.max_seq = max_seq
    self.num_layers = num_layers

  def forward(self, x, return_states = False): # x : 모든 배치의 모든 시점의 단어 임베딩 벡터 (batch_size, max_len, dim_embed)
    input_list = x
    max_seq = x.size()[1]
    h_output = []
    for floor in range(self.num_layers): # 양방향 모델로 만들 경우 이전 층의 양 방향 hidden state를 입력값으로 받기 위해 forward 메소드 내에 시점 반영
      h_output_layer = []
      h_forward = self.forward_layers[floor].forward(input_list, max_seq)
      h_output_layer.append(h_forward)
      if self.direction == 2:        
        h_backward = self.backward_layers[floor].forward(input_list, max_seq)
        h_output_layer.append(h_backward)
      h_output.append(h_output_layer)
      input_list = torch.cat(h_output_layer, dim = 2)
    self.states = torch.cat(h_output_layer, dim = 2)
    return self.states, self.states[:, -1, :] # 마지막 시점의  hidden state를 따로 뽑음

    
class attention(nn.Module):
  def __init__(self, dim_model, batch_size):
    super().__init__()
    self.W = nn.Linear(2*dim_model, dim_model)
    self.U = nn.Linear(2*dim_model, dim_model)
    self.v = nn.Linear(dim_model, 1)
    self.batch_size = batch_size

  def get_attention(self, hidden_state, encoding_matrix): # hidden state : (batch_size, 2*dim_model), encoding_matrix : (bath_size, max_seq, 2*dim_model)
    if len(hidden_state.size()) == 2 :
      hidden_state = hidden_state.unsqueeze(1) # (batch_size, 1, 2*dim_model)
    max_seq = encoding_matrix.size()[1]
    attention_score = self.v(torch.tanh(self.W(hidden_state) + self.U(encoding_matrix))) # (batch_size, max_seq, 1)
    attention_dist = torch.softmax(attention_score, dim = 1) # (batch_size, max_seq, 1)
    attention_vect = (attention_dist.expand(self.batch_size, max_seq, 2*dim_model)*encoding_matrix).sum(dim = 1)
    return attention_vect


class Decoder(nn.Module):
  def __init__(self, dim_model, dim_embed, batch_size, encoder_bidirectional = True):
    super().__init__()
    self.direction = 2 if encoder_bidirectional else 1
    dim_decoder = self.direction*dim_model
    self.W_z = nn.Linear(dim_embed, dim_decoder)
    self.W_r = nn.Linear(dim_embed, dim_decoder)
    self.W_s = nn.Linear(dim_embed, dim_decoder)

    self.U_z = nn.Linear(2*dim_model, dim_decoder)
    self.U_r = nn.Linear(2*dim_model, dim_decoder)
    self.U_s = nn.Linear(2*dim_model, dim_decoder)

    self.V_z = nn.Linear(2*dim_model, dim_decoder)
    self.V_r = nn.Linear(2*dim_model, dim_decoder)
    self.V_s = nn.Linear(2*dim_model, dim_decoder)

    self.attention = nn.ModuleList([attention(dim_model, batch_size)])

  def forward(self, x, h_old, encoding_matrix): # x : 모든 배치의 이전 시점 출력 단어 임베딩 벡터 (batch_size, dim_embed) / h_old 이전 시점 hidden state (batch_size, 2*dim_model)
    attention_vect = self.attention[0].get_attention(h_old, encoding_matrix) # (batch_size, 2*dim_embed)
    r = torch.sigmoid(self.W_r((x)) + self.U_r(h_old) + self.V_r(attention_vect))
    z = torch.sigmoid(self.W_z((x)) + self.U_z(h_old) + self.V_z(attention_vect))
    h_new = torch.tanh(self.W_s((x)) + self.U_s(r*h_old) + self.V_s(attention_vect))
    h = (1 - z)*h_old + z*h_new
    return h