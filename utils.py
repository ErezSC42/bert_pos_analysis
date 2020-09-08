import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import contractions
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder


def tokenize_word(sentence_ids, target_word, bert_tokenizer):

    word_mask = len(sentence_ids) * [0]
    word_ids = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(target_word))
    try:
        word_ids_indexes_in_text = [sentence_ids.index(word) for word in word_ids]
        for tok_idx in word_ids_indexes_in_text:
            word_mask[tok_idx] = 1
        return word_mask
    except:
        return word_mask

def preprocess_text(x: str, tokenizer: BertTokenizer, max_sequence_len: int):
    cur_x = x
    cur_x = "[CLS] " + cur_x
    cur_x = cur_x.replace("\n", "")
    cur_x = cur_x.replace(" cannot ", " can not ")
    cur_x = tokenizer.tokenize(cur_x)
    cur_x = tokenizer.convert_tokens_to_ids(cur_x)
    cur_x = cur_x[:max_sequence_len]
    cur_x = cur_x + [0] * (max_sequence_len - len(cur_x))
    return cur_x



def extract_attn_mask(x: list, max_sequence_len):
    first_0_token_idx = x.index(0)
    return first_0_token_idx * [1] + (max_sequence_len - first_0_token_idx) * [0]

def text_to_dataloader(
        sentences_df: pd.DataFrame,
        device: torch.device,
        inference_batch_size: int,
        bert_tokenizer: BertTokenizer,
        max_sequence_len: int) -> DataLoader:
    '''
    mutates dataframe!
    :param sentenecs: pd.DataFrame,
    :return: returns a torch.utils.data.Dataloader objects that iterates over the input data
    '''
    assert isinstance(sentences_df, pd.DataFrame)
    assert "text" in sentences_df.columns
    assert "label" in sentences_df.columns

    sentences_df["label"] = sentences_df["label"].astype("category")
    sentences_df["label_idx"] = sentences_df["label"].cat.codes
    sentences_df["text_ids"] = sentences_df["text"].apply(lambda x: preprocess_text(x, bert_tokenizer,max_sequence_len))
    sentences_df["attn_mask"] = sentences_df["text_ids"].apply(lambda x: extract_attn_mask(x, max_sequence_len))
    sentences_df["query_mask"] = sentences_df.apply(lambda row: tokenize_word(row.text_ids, row.word, bert_tokenizer), axis=1)

    # TODO add target word mask extraction

    sentences_idx_tensor = torch.LongTensor(np.stack(sentences_df["text_ids"].values)).to(device)
    sentences_mask_tensor = torch.LongTensor(np.stack(sentences_df["attn_mask"].values)).to(device)

    # build dataset
    inference_tensor_dataset = TensorDataset(
        sentences_idx_tensor.to(device=device),
        sentences_mask_tensor.to(device=device)
    )

    # build dataloader
    inference_dataloader = DataLoader(inference_tensor_dataset, batch_size=inference_batch_size)
    return inference_dataloader



def check_accuracy_classification(data_loader: DataLoader,model,name, total, criterion=None, verbose=True, use_masks=True, drop_grad=True):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data in data_loader:
           if use_masks:
               text, mask, labels = data
               outputs, _ = model(text, mask)
           else:
               text, labels = data
               outputs = model(text)
           real_labels = torch.max(labels, 1)[1]
           _, predicted = torch.max(outputs.data, 1)
           correct += (predicted == real_labels).sum().item()
           if criterion is not None:
               loss = criterion(outputs, real_labels)
        loss /= total
        acc = correct / total
        if name == "train" and verbose:
            print(f'\t Train Loss: {loss:.3f} | Acc: {acc * 100:.2f}% | Correct: {correct}/{total} ')
        elif verbose:
            print(f'\t  Val. Loss: {loss:.3f} | Acc: {acc * 100:.2f}% | Correct: {correct}/{total} ')
        return loss, acc


def calc_performance(model : torch.nn.Module, dloader: DataLoader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in tqdm.tqdm(dloader):
            text, mask, labels = data
            outputs = model(text, mask)
            _, predicted = torch.max(outputs[0], 1)
            label = torch.max(labels, 1)[1].cpu().detach().tolist()
            y_pred = y_pred + predicted.tolist()
            y_true = y_true + label
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return y_true, y_pred