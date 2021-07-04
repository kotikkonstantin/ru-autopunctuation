from datetime import datetime
import json
import numpy as np
import os
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoTokenizer
import warnings

from base import init_random_seed_torch, gpus_to_use, get_device
from data import load_file, preprocess_data, create_data_loader
from model import BertPunc

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def validate(model, criterion, epoch, epochs, iteration, iterations, data_loader_valid, save_path, train_loss,
             best_val_loss, best_model_path, punctuation_enc):
    val_losses = []
    val_accs = []
    val_f1s = []

    label_keys = list(punctuation_enc.keys())
    label_vals = list(punctuation_enc.values())

    for inputs, labels in tqdm(data_loader_valid, total=len(data_loader_valid)):
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            val_loss = criterion(output, labels)
            val_losses.append(val_loss.cpu().data.numpy())

            y_pred = output.argmax(dim=1).cpu().data.numpy().flatten()
            y_true = labels.cpu().data.numpy().flatten()
            val_accs.append(metrics.accuracy_score(y_true, y_pred))
            val_f1s.append(metrics.f1_score(y_true, y_pred, average=None, labels=label_vals))

    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)
    val_f1 = np.array(val_f1s).mean(axis=0)

    improved = ''

    # model_path = '{}model_{:02d}{:02d}'.format(save_path, epoch, iteration)
    model_path = save_path + 'model'
    torch.save(model.state_dict(), model_path)
    if val_loss < best_val_loss:
        improved = '*'
        best_val_loss = val_loss
        best_model_path = model_path

    f1_cols = ';'.join(['f1_' + key for key in label_keys])

    progress_path = save_path + 'progress.csv'
    if not os.path.isfile(progress_path):
        with open(progress_path, 'w') as f:
            f.write('time;epoch;iteration;training loss;loss;accuracy;' + f1_cols + '\n')

    f1_vals = ';'.join(['{:.4f}'.format(val) for val in val_f1])

    with open(progress_path, 'a') as f:
        f.write('{};{};{};{:.4f};{:.4f};{:.4f};{}\n'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch + 1,
            iteration,
            train_loss,
            val_loss,
            val_acc,
            f1_vals
        ))

    print("Epoch: {}/{}".format(epoch + 1, epochs),
          "Iteration: {}/{}".format(iteration, iterations),
          "Loss: {:.4f}".format(train_loss),
          "Val Loss: {:.4f}".format(val_loss),
          "Acc: {:.4f}".format(val_acc),
          "F1: {}".format(f1_vals),
          improved)

    return best_val_loss, best_model_path


def train(model, optimizer, criterion, epochs, data_loader_train, data_loader_valid, save_path, punctuation_enc,
          iterations=3, best_val_loss=1e9):
    print_every = len(data_loader_train) // iterations + 1
    best_model_path = None
    model.train()
    pbar = tqdm(total=print_every)

    for e in range(epochs):

        counter = 1
        iteration = 1

        for inputs, labels in data_loader_train:

            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss = loss.cpu().data.numpy()

            pbar.update()

            if counter % print_every == 0:
                pbar.close()
                model.eval()
                best_val_loss, best_model_path = validate(model, criterion, e, epochs, iteration, iterations,
                                                          data_loader_valid,
                                                          save_path, train_loss, best_val_loss, best_model_path,
                                                          punctuation_enc)
                model.train()
                pbar = tqdm(total=print_every)
                iteration += 1

            counter += 1

        pbar.close()
        model.eval()
        best_val_loss, best_model_path = validate(model, criterion, e, epochs, iteration, iterations, data_loader_valid,
                                                  save_path, train_loss, best_val_loss, best_model_path,
                                                  punctuation_enc)
        model.train()
        if e < epochs - 1:
            pbar = tqdm(total=print_every)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    return model, optimizer, best_val_loss


if __name__ == '__main__':

    init_random_seed_torch(1995)
    gpus_list = [0, 1, 2, 4]
    gpus_to_use(gpus_list)
    device = get_device(gpus_list)

    punctuation_enc = {
        'O': 0,
        'COMMA': 1,
        'PERIOD': 2,
        'QUESTION': 3
    }

    segment_size = 16
    dropout = 0.3
    epochs_top = 1
    iterations_top = 2
    batch_size_top = 512
    learning_rate_top = 1e-5
    epochs_all = 4
    iterations_all = 3
    batch_size_all = 256
    learning_rate_all = 1e-5

    hyperparameters = {
        'segment_size': segment_size,
        'epochs_top': epochs_top,
        'iterations_top': iterations_top,
        'batch_size_top': batch_size_top,
        'learning_rate_top': learning_rate_top,
        'epochs_all': epochs_all,
        'iterations_all': iterations_all,
        'batch_size_all': batch_size_all,
        'learning_rate_all': learning_rate_all,
        'model': {
            "name_or_path": 'DeepPavlov/rubert-base-cased-sentence',
            'dropout': dropout,
            "mode": {
                "name": "stacked_hidden_states",
                "config": {
                    "type": "concat",
                    "n_layers": 4,
                    "reverse": True,
                    "sent_agg_type": "mean"
                }
            }
        }
    }

    print('LOADING DATA...')
    data_train = load_file('datasets/train_echomsk.txt')
    data_valid = load_file('datasets/valid_echomsk.txt')
    print(f"Train shape: {len(data_train)}, Valid shape: {len(data_valid)}")

    tokenizer = AutoTokenizer.from_pretrained(hyperparameters['model']['name_or_path'], do_lower_case=True)
    hyperparameters['model']['tokenizer_vocab_size'] = tokenizer.vocab_size

    save_path = 'checkpoints/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.mkdir(save_path)
    with open(save_path + 'hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)

    print('PREPROCESSING DATA...')
    X_train, y_train = preprocess_data(data_train, tokenizer, punctuation_enc, segment_size)
    X_valid, y_valid = preprocess_data(data_valid, tokenizer, punctuation_enc, segment_size)

    print('INITIALIZING MODEL...')
    output_size = len(punctuation_enc)
    bert_punc = nn.DataParallel(BertPunc(segment_size, output_size, hyperparameters['model']).to(device))

    print('TRAINING TOP LAYER...')
    data_loader_train = create_data_loader(X_train, y_train, True, batch_size_top)
    data_loader_valid = create_data_loader(X_valid, y_valid, False, batch_size_top)
    for p in bert_punc.module.bert.parameters():
        p.requires_grad = False
    optimizer = optim.Adam(bert_punc.parameters(), lr=learning_rate_top)
    criterion = nn.CrossEntropyLoss()
    bert_punc, optimizer, best_val_loss = train(bert_punc, optimizer, criterion, epochs_top,
                                                data_loader_train, data_loader_valid, save_path, punctuation_enc,
                                                iterations_top, best_val_loss=1e9)

    print('TRAINING ALL LAYERS...')
    data_loader_train = create_data_loader(X_train, y_train, True, batch_size_all)
    data_loader_valid = create_data_loader(X_valid, y_valid, False, batch_size_all)
    for p in bert_punc.module.bert.parameters():
        p.requires_grad = True
    optimizer = optim.Adam(bert_punc.parameters(), lr=learning_rate_all)
    criterion = nn.CrossEntropyLoss()
    bert_punc, optimizer, best_val_loss = train(bert_punc, optimizer, criterion, epochs_all,
                                                data_loader_train, data_loader_valid, save_path, punctuation_enc,
                                                iterations_all, best_val_loss=best_val_loss)
