import torch
from torch import nn
from transformers import BertModel, BertConfig


class BertPunc(nn.Module):

    def __init__(self, segment_size, output_size, config):
        super(BertPunc, self).__init__()

        model_config = BertConfig.from_pretrained(config['name_or_path'])
        model_config.output_hidden_states = True

        self.bert = BertModel.from_pretrained(config['name_or_path'], config=model_config)
        self.bert_vocab_size = config['tokenizer_vocab_size']
        self.config = config

        if config['mode']['name'] == 'lm_logits':
            self.bn = nn.BatchNorm1d(segment_size * self.bert_vocab_size)
            self.fc = nn.Linear(segment_size * self.bert_vocab_size,
                                output_size)
        elif config['mode']['name'] == 'stacked_hidden_states':
            if config['mode']['config']['type'] == "concat":
                self.fc = nn.Linear(model_config.hidden_size * config['mode']['config']['n_layers'], output_size)

        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.bert(x)

        if self.config['mode']['name'] == 'lm_logits':
            x = x.logits
            x = x.view(x.shape[0], -1)
            x = self.fc(self.dropout(self.bn(x)))

        elif self.config['mode']['name'] == 'stacked_hidden_states':
            x = x.hidden_states

            _reverse = -1 if self.config['mode']['config']['reverse'] else 1

            x = torch.cat(tuple(x[_reverse * i] for i in range(1, self.config['mode']['config']['n_layers'] + 1)), dim=-1)

            if self.config['mode']['config']['sent_agg_type'] == "mean":
                x = x.mean(dim=1)

            x = self.dropout(x)
            x = self.fc(x)

        return x
