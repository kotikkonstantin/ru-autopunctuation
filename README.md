# Motivation
Speech Recognition (ASR) systems usually output unpunctuated and uncased texts (transcripts). The punctuation restoration improves the readability of ASR transcripts.

This work is about punctuation restoration for **the Russian language**. 
  
As a backbone approach, it's used [BertPunc](https://github.com/nkrnrnk/BertPunc) project. It's the state-of-the-art approach for English language punctuation restoration (at the time of publication).

As a backbone Transformer-based model, it's used [DeepPavlov/rubert-base-cased-sentence](https://huggingface.co/DeepPavlov/rubert-base-cased-sentence).

The supported punctuation signs: **period, comma, and question.**

# New made things with the comparison with [BertPunc](https://github.com/nkrnrnk/BertPunc):
+ direct decoding (to get human-readable transcript)
+ use embeddings from Transformer-based encoder layers instead of use LM logits (to decrease dimension in front of FC)

# Repository structure

------------------------
```
    .
    ├── checkpoints          <- folder to keep experiments artifacts
    ├── datasets             <- folder to keep train/valid/test datasets
    ├── base.py              <- auxiliary functions: initialize random seed, device choice 
    ├── data.py              <- make target, prepare datasets for training
    ├── model.py             <- form model architecture 
    ├── train.py             <- train-eval loop
    ├── inference.py         <- perfom punctuation restored transcripts
    └── README.md
```

##  Dataset preparing
1. For model training, it's used radio dialogues data (provided by [link](https://github.com/vadimkantorov/convasr))

2. The script `make_datasets.py`:
    + takes raw or preprocessed text corpus
    + splits on *train/valid/test*
    + transforms every splits to format:
    
    ![](.README_images/1.png)
    
(if after token there is a period, comma, or question sign then the token is annotated with the corresponding label)
       
4. Dataset stats:
+ Train  - 6 761 752 annotated tokens
+ Valid  - 2 135 743 annotated tokens
+ Test - 116 853 annotated tokens


## Modeling
Used hyperparameters you can find in *checkpoints* folder.

### Metrics values on the validation dataset
 ![](.README_images/5.png)

### Metrics values on test dataset by best validation loss checkpoint
![](.README_images/6.png)

## Let's see how it restores data transcripts:

+ Example input from test data:
[ ![](.README_images/2.png ) ](.README_images/2.png )


+ Punctuation and casing restored: 
[ ![](.README_images/3.png ) ](.README_images/3.png )


+ Ideal GT:
[ ![](.README_images/4.png ) ](.README_images/4.png )


## Let's see how it restores non-domain data (zero-shot learning)
It will be a medical text example (provided by [N.N. Burdenko Neurosurgery National Medical Research Center](https://www.nsi.ru/)) 
+ Example input:
[ ![](.README_images/7.png ) ](.README_images/7.png )

+ Punctuation and casing restored: 
[ ![](.README_images/8.png ) ](.README_images/8.png )


+ Ideal GT:
[ ![](.README_images/9.png ) ](.README_images/9.png )


### See how it looks when it was fine-tuned on Russian medical abstracts of ["РИНЦ"](https://elibrary.ru/) papers
This data also was provided by  [N.N. Burdenko Neurosurgery National Medical Research Center](https://www.nsi.ru/)

**See results/comparison_results_table.xlsx**