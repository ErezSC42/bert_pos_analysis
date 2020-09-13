import json
import tqdm
import torch
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, confusion_matrix


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

    LABELS_TO_DROP = ["X", "_"]
    df = sentences_df[~sentences_df["label"].isin(LABELS_TO_DROP)]
    df["label"] = df["label"].astype("category")
    with open("pos_to_label.json", "rb") as fp:
        pos_to_label_dict = json.load(fp)

    df["label_idx"] = df["label"].map(pos_to_label_dict)
    df["text_ids"] = df["text"].apply(lambda x: preprocess_text(x, bert_tokenizer,max_sequence_len))
    df["attn_mask"] = df["text_ids"].apply(lambda x: extract_attn_mask(x, max_sequence_len))
    df["query_mask"] = df.apply(lambda row: tokenize_word(row.text_ids, row.word, bert_tokenizer), axis=1)

    sentences_idx_tensor = torch.LongTensor(np.stack(df["text_ids"].values)).to(device)
    sentences_mask_tensor = torch.LongTensor(np.stack(df["attn_mask"].values)).to(device)
    query_mask_tensor = torch.LongTensor(np.stack(df["query_mask"].values)).to(device)
    label_tensor = torch.LongTensor(np.stack(df["label_idx"].values)).to(device)

    # build dataset
    inference_tensor_dataset = TensorDataset(
        sentences_idx_tensor.to(device=device),
        sentences_mask_tensor.to(device=device),
        query_mask_tensor,
        label_tensor
    )

    # build dataloader
    inference_dataloader = DataLoader(inference_tensor_dataset, batch_size=inference_batch_size)
    return df, inference_dataloader



def plot_confusion_matrix(
        y_true,
        y_pred,
        normalize=False,
        cmap=plt.cm.Blues,
        label_list = None,
        visible=True,
        savepath=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="micro")
    title = f"Confusion Matrix, Acc: {acc:.2f}, F1: {f1:.2f}"


    if label_list == None:
        classes = range(0, max(y_true))
    else:
        classes = label_list
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure(figsize=(13,13))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if savepath is not None:
        plt.savefig(savepath)
    if visible:
        plt.show()
    return acc, f1



def check_accuracy_classification(data_loader: DataLoader,model,name, total, criterion=None, verbose=True):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data in data_loader:
            text, mask, target_masks, labels = data
            outputs = model(text, mask, target_masks)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            if criterion is not None:
                loss = criterion(outputs, labels)
        loss /= total
        acc = correct / total
        if name == "train" and verbose:
            print(f'\t Train Loss: {loss:.3f} | Acc: {acc * 100:.2f}% | Correct: {correct}/{total} ')
        elif verbose:
            print(f'\t  Val. Loss: {loss:.3f} | Acc: {acc * 100:.2f}% | Correct: {correct}/{total} ')
        return loss, acc


def calc_classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, precision, recall, f1


def calc_performance(model : torch.nn.Module, dloader: DataLoader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in tqdm.tqdm(dloader):
            text, mask, target_masks, labels = data
            outputs = model(text, mask, target_masks)
            _, predicted = torch.max(outputs.data, 1)
            label = labels.cpu().detach().tolist()
            y_pred = y_pred + predicted.tolist()
            y_true = y_true + label
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return y_true, y_pred