import re
from sklearn.model_selection import train_test_split
import string
from typing import Dict, List, Optional, Text, Tuple

TOKEN_RE = re.compile(r'-?\d*\.\d+|[a-zа-яё]+|-?\d+|\S', re.I)

punctuation_enc = {
    'O': 0,
    'COMMA': 1,
    'PERIOD': 2,
    'QUESTION': 3
}


def tokenize_text_simple_regex(txt: str, regex: re.Pattern, min_token_size: int = 0) -> List[str]:
    """Tokenize text with simple regex
    Args:
        txt: text to tokenize
        regex: re.compile output
        min_token_size: min char length to highlight as token

    Returns:
        tokens list
    """

    txt = txt.lower()
    all_tokens = regex.findall(txt)
    return [token for token in all_tokens if len(token) >= min_token_size]


def tokenize(corpus: List[str]) -> List[List[str]]:
    """Tokenize text corpus with simple regex
    Args:
        corpus: text corpus
    Returns:
        List of tokenized texts
    """
    tokenized_corpus = []
    for doc in corpus:
        tokenized_corpus.append(tokenize_text_simple_regex(doc, TOKEN_RE))

    return tokenized_corpus


def make_labeling(tokenized_corpus: List[List[str]], save_path: Optional[str] = None) -> List[List[str]]:
    """
    Make labeling to correspond BertPunc input data https://github.com/IsaacChanghau/neural_sequence_labeling/tree/master/data/raw/LREC
    Args:
        tokenized_corpus: tokenized text corpus
        save_path: path to save labeling result
    Returns:
        labeled tokenized text corpus
    """
    labeled_tokens = []
    for text_tokenized in tokenized_corpus:
        text_tokenized.append("")
        for i in range(len(text_tokenized) - 1):
            if text_tokenized[i] in string.punctuation:
                if text_tokenized[i + 1] == ".":
                    labeled_tokens[-1][1] = "PERIOD"
                elif text_tokenized[i + 1] == ",":
                    labeled_tokens[-1][1] = "COMMA"
                elif text_tokenized[i + 1] == "?":
                    labeled_tokens[-1][1] = "QUESTION"
                else:
                    continue
            else:
                if text_tokenized[i + 1] == ".":
                    labeled_tokens.append([text_tokenized[i], "PERIOD"])
                elif text_tokenized[i + 1] == ",":
                    labeled_tokens.append([text_tokenized[i], "COMMA"])
                elif text_tokenized[i + 1] == "?":
                    labeled_tokens.append([text_tokenized[i], "QUESTION"])
                else:
                    labeled_tokens.append([text_tokenized[i], "O"])

    if save_path is not None:
        with open(save_path, "w") as f:
            for token, label in labeled_tokens:
                f.write(f"{token}\t{label}\n")

    return labeled_tokens


def make_datasets(path_to_preprocessed_corpus: Text, config: Dict) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """
    Create labelled train, valid, test datasets for BertPunc
    Args:
        path_to_preprocessed_corpus: path to preprocessed text corpus
        config: config
    Returns:
        (train corpus, valid corpus, test corpus)
    """

    with open(path_to_preprocessed_corpus) as f:
        corpus = f.readlines()

    train_corpus, valid_corpus = train_test_split(corpus, random_state=config['random_seed'],
                                                  test_size=config['valid_rate'] + config['test_rate'])
    valid_corpus, test_corpus = train_test_split(valid_corpus, random_state=config['random_seed'],
                                                 test_size=config['test_rate'])

    train_corpus = tokenize(train_corpus)
    valid_corpus = tokenize(valid_corpus)
    test_corpus = tokenize(test_corpus)

    train_corpus = make_labeling(train_corpus, config['train_path_name'])
    valid_corpus = make_labeling(valid_corpus, config['valid_path_name'])
    test_corpus = make_labeling(test_corpus, config['test_path_name'])

    print(f"Train amount: {len(train_corpus)}.\nValid amount: {len(valid_corpus)}.\nTest amount: {len(test_corpus)}")

    return train_corpus, valid_corpus, test_corpus


