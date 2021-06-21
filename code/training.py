import time
import copy
import matplotlib.pyplot as plt
import torch
import numpy as np

class early_stopping:
  def __init__(self, patience = 10, save_path = "./save_model"):
    self.patience = patience
    self.save_path = save_path
    self.count = 0
    self.best_score = np.Inf
    self.stop = False
    self.best_model = None

  def __call__(self, val_loss, model):
    if self.best_score == None:
      self.best_score = val_loss
      self.save_model(model)
    elif val_loss < self.best_score:
      self.best_score = val_loss
      self.save_model(model)
      self.count = 0
      print("new best model is saved")
      self.best_model = copy.deepcopy(model)
    else:
      self.count += 1
      if self.count == self.patience:
        print('-'*50)
        print(f"training is over")
        print('-'*50)
        self.stop = True

  def save_mode(self, model):
    torch.save(model.state_dict(), self.save_path)


def train_per_epoch(model, iterator, optimizer, criterion, clip, device):
  model.train()
  epoch_loss = 0
  for enu, (target_lang, source_lang) in enumerate(iterator):
    source_lang, target_lang = source_lang.to(device), target_lang.to(device)
    optimizer.zero_grad()
    output = model(source_lang, target_lang)

    output = output[1:].view(-1, output.size()[-1]).to(device)
    target_lang = target_lang[1:].view(-1)
    loss = criterion(output, target_lang)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
    optimizer.step()
    
    epoch_loss += loss.item()
    print(f"loss of {enu} enumerate is {loss:.3f}")

  return epoch_loss/len(iterator)


def evaluate(model, iterator, criterion, device):
  model.eval()
  epoch_loss = 0 
  with torch.no_grad():
    for enu, (target_lang, source_lang) in enumerate(iterator):
      source_lang, target_lang = source_lang.to(device), target_lang.to(device)

      output = model(source_lang, target_lang, teach_forcing_ratio = 0)

      output = output[1:].view(-1, output.size()[-1]).to(device)
      target_lang = target_lang[1:].view(-1)

      loss = criterion(output, target_lang)

      epoch_loss += loss.item()
    return epoch_loss/len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(num_epochs, clip, model, train_iter, dev_iter, test_iter, criterion, device):
  os.chdir("./training_log")
  train_loss_log = []
  valid_loss_log = []
  early_stopper = early_stopping(patience = 10, save_path = "./save_model")
  for epoch in range(num_epochs):
    start_time = time.time()

    train_loss = train_per_epoch(model, train_iter, optimizer, criterion, clip, device)
    valid_loss = evaluate(model, dev_iter, criterion, device)
    early_stopper(valid_loss, model)
    train_loss_log.append(train_loss)
    valid_loss_log.append(valid_loss)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    if early_stopper.stop:
      break
  test_loss = evaluate(early_stopper.best_model, test_iter, criterion, device)

  plt.plot(train_loss_log)
  plt.plot(valid_loss_log)
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'dev'], loc='upper left')
  plt.savefig(f"./training_fig.png", )

  print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')