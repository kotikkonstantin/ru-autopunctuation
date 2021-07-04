from functools import partial
import json
import re
from pkg_resources import resource_filename
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer

from base import init_random_seed_torch, gpus_to_use, get_device
from data import preprocess_data, create_data_loader
from make_datasets import tokenize, make_labeling
from model import BertPunc


init_random_seed_torch(1995)


labels_mapper = {
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTION": "?",
    "O": ""
}

punctuation_enc = {
    'O': 0,
    'COMMA': 1,
    'PERIOD': 2,
    'QUESTION': 3
}

inv_punctuation_enc = {v: k for k, v in punctuation_enc.items()}


def predictions(data_loader, bert_punc, device):
    y_pred = []
    y_true = []
    for inputs, labels in tqdm(data_loader, total=len(data_loader), disable=True):
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            output = bert_punc(inputs)
            y_pred += list(output.argmax(dim=1).cpu().data.numpy().flatten())
            y_true += list(labels.cpu().data.numpy().flatten())
    return y_pred, y_true


def decode_gt(labeled_tokens: list, labels_mapper: dict):
    result = []
    for item in labeled_tokens:
        token, label = item.split("\t")
        token, label = token.strip(), label.strip()
        result.append(token + labels_mapper[label])
    return result


def right_decode_predictions(data_test, predictions, tokenizer, punctuation_enc, segment_size):
    """It takes linear time to execute
    """

    temp_X_test, _ = preprocess_data(data_test, tokenizer, punctuation_enc, segment_size)
    temp_X_test_decoded = []
    temp_X_test_encoded = []

    substitution = {0: "", 1: ',', 2: '.', 3: '?'}

    index_count = 3

    merged_encoded = []

    for i, (encoded_str, pred_y) in enumerate(zip(temp_X_test, predictions)):

        encoded_str = encoded_str[((segment_size - 1) // 2 - 1):]
        encoded_str = encoded_str[:segment_size // 2]
        encoded_str = encoded_str.tolist()
        _s = tokenizer.decode(encoded_str)

        _s = _s.replace(" [PAD]", substitution[pred_y])

        temp_X_test_decoded.append(_s)
        temp_X_test_encoded.append(encoded_str)

        if i == 0:
            merged_encoded += encoded_str
            continue

        merged_encoded.insert(index_count, 0)
        index_count += 2

        # TODO REVISE (problem with SEP and CLS token)
        #     if pred_y == 0:
        #         l1.insert(index_count, 0)
        #         index_count += 2
        #     else:
        #         enc_punct = tokenizer.encode(substitution[pred_y])
        #         l1[index_count: index_count + len(enc_punct) + 1] = enc_punct
        #         index_count += 2 + len(enc_punct)

        merged_encoded.append(encoded_str[-1])

    final_s = tokenizer.decode(merged_encoded)

    final_s_pad_tokenized = final_s.split(" [PAD]")

    filled_final_s = ""
    for i in range(len(predictions[:len(final_s_pad_tokenized)])):
        filled_final_s += final_s_pad_tokenized[i] + substitution[predictions[i]]

    filled_final_s = re.sub(r"< empty >\W", "", filled_final_s)
    return filled_final_s


def make_single_text_pred(domain_text, func_to_pred, segment_size, batch_size,
                          tokenizer, punctuation_enc, device):
    _prepared_domain_text = make_labeling(tokenize([domain_text]))
    prepared_domain_text = []
    for token, label in _prepared_domain_text:
        prepared_domain_text.append(f"{token}\t{label}\n")

    if len(prepared_domain_text) < segment_size:
        prepared_domain_text = prepared_domain_text[:] + ["<empty>\tO\n"] * (segment_size - len(prepared_domain_text))
        X_domain, y_domain = preprocess_data(prepared_domain_text, tokenizer, punctuation_enc, segment_size)
        data_loader_one_shot = create_data_loader(X_domain, y_domain, False, batch_size)
        y_pred_domain, _ = func_to_pred(data_loader_one_shot)
    else:
        X_domain, y_domain = preprocess_data(prepared_domain_text, tokenizer, punctuation_enc, segment_size)
        data_loader_one_shot = create_data_loader(X_domain, y_domain, False, batch_size)
        y_pred_domain, _ = func_to_pred(data_loader_one_shot)

    return prepared_domain_text, y_pred_domain


def capitalize(text: str) -> str:
    text = text.strip()
    if len(text) == 0:
        return ""

    text = text.capitalize()
    splitted_text = re.split("([?.]\s*)", text)
    splitted_text = [substring.capitalize() for substring in splitted_text]
    capitalized_text = "".join(splitted_text)

    return capitalized_text


def cnt_punct(s):
    count = 0
    for i in range(0, len(s)):
        # Checks whether given character is a punctuation mark
        if s[i] in ('!', ",", "\'", ";", "\"", ".", "-", "?"):
            count = count + 1

    return count


def model_and_tokenizer_initialize(device, hyperparameters: dict):
    model_name = hyperparameters['model']['name_or_path']

    output_size = len(punctuation_enc)

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    bert_punc = nn.DataParallel(BertPunc(hyperparameters['segment_size'],
                                         output_size,
                                         hyperparameters['model']).to(device))

    bert_punc.load_state_dict(torch.load(get_path_to_checkpoint(), map_location=device))

    bert_punc.eval()

    return bert_punc, tokenizer


def inference(input_text: str, bert_punc, tokenizer, device, hyperparameters: dict, batch_size: int = 2048):
    input_text = input_text.strip()

    func_to_pred = partial(predictions, bert_punc=bert_punc, device=device)

    prepared_domain_text, y_pred_domain = make_single_text_pred(input_text,
                                                                func_to_pred,
                                                                hyperparameters['segment_size'],
                                                                batch_size,
                                                                tokenizer,
                                                                punctuation_enc,
                                                                device
                                                                )

    res = right_decode_predictions(prepared_domain_text, y_pred_domain,
                                   tokenizer,
                                   punctuation_enc,
                                   hyperparameters["segment_size"])

    res = res[:len(input_text) + cnt_punct(res)]

    res = capitalize(res)
    return res


def prepare_hyperparameters():
    path = "checkpoints/echomsk6000/20210430_125222_rubert-base-cased-sentence_segment_size_16_stacked_hidden_states_concat_4_mean_sent_agg/hyperparameters.json"
    path = resource_filename(__name__, path)
    with open(path, 'r') as f:
        hyperparameters = json.load(f)

    return hyperparameters


def get_path_to_checkpoint():
    path = "checkpoints/echomsk6000/20210430_125222_rubert-base-cased-sentence_segment_size_16_stacked_hidden_states_concat_4_mean_sent_agg/model"
    path = resource_filename(__name__, path)
    return path


if __name__ == '__main__':

    BATCH_SIZE = 64

    gpus_list = [2]
    gpus_to_use(gpus_list)
    device = get_device(gpus_list)

    hyperparameters = prepare_hyperparameters()

    bert_punc, tokenizer = model_and_tokenizer_initialize(device, hyperparameters)

    input_text = "они ухнуты известно куда отчасти эта система выгодна и для производителей кино потому что она " \
                 "позволяет им не заботиться о кассовых сборах потому что все деньги зарабатываются на этапе " \
                 "выделения финансирования и так далее но болезнь зашла к сожалению так далеко что взять просто и " \
                 "перевести всех на подножный корм или на самофинансирование это значит просто все убить это надо " \
                 "медленномедленно лечить хвост по частям и наверное я бы начал это лечение с формирования просто " \
                 "нескольких профессиональных советов из уважаемых в своей профессиональной сфере людей которые бы не " \
                 "контролировали репертуар а с которыми бы советовались принимая решение о том или ином " \
                 "госфинансировании то есть разделить это единое туловище минкульта на несколько таких корпоративных " \
                 "профессиональных гильдий"

    print(input_text)

    punc_case_restored_text = inference(input_text,bert_punc, tokenizer, device, hyperparameters, BATCH_SIZE)

    print(punc_case_restored_text)