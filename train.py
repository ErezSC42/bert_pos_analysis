import os
import json
import torch
import pandas as pd
import transformers
import torch.nn as nn
from torch.optim import Adam
from models import BertProbeClassifer
from utils import check_accuracy_classification
from utils import text_to_dataloader, plot_confusion_matrix, calc_performance


MATRICES_SAVE_PATH = os.path.join("confusion_matrices")

def txt_to_dataframe(data_path):
    '''
    read UD text file and convert to df format
    '''
    with open(data_path, "r") as fp:
        df = pd.DataFrame(
            columns={
                "text",
                "word",
                "label"
            }
        )
        for line in fp.readlines():
            if TEXT_CONST in line:
                words_list = []
                labels_list = []
                text = line.split(TEXT_CONST)[1]
                # this is a new text, need to parse all the words in it
            elif line is not STOP_CONST and HEADER_CONST not in line:
                temp_list = line.split("\t")
                words_list.append(temp_list[WORD_OFFSET])
                labels_list.append(temp_list[LABEL_OFFSET])
            if line == STOP_CONST:
                # this is the end of the text, adding to df
                cur_df = pd.DataFrame(
                    {
                        "text": len(words_list) * [text],
                        "word": words_list,
                        "label": labels_list
                    }
                )
                df = pd.concat([df, cur_df])
        return df


if __name__ == '__main__':
    train_path = os.path.join("data", "en_partut-ud-train.conllu")
    dev_path = os.path.join("data", "en_partut-ud-dev.conllu")
    test_path = os.path.join("data", "en_partut-ud-test.conllu")

    HEADER_CONST = "# sent_id = "
    TEXT_CONST = "# text = "
    STOP_CONST = "\n"
    WORD_OFFSET = 1
    LABEL_OFFSET = 3

    df_train = txt_to_dataframe(train_path)
    df_dev = txt_to_dataframe(dev_path)
    df_test = txt_to_dataframe(test_path)

    train_len = len(df_train)
    test_len = len(df_dev)

    bert_tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    batch_size = 128
    epochs = 5
    classes_count = 16
    num_hidden_layers = 1

    df_train = df_train.head(train_len)
    df_dev = df_dev.head(test_len)

    df_train, dataloader_train = text_to_dataloader(df_train, "cuda", batch_size, bert_tokenizer, 256)
    df_dev, dataloader_dev = text_to_dataloader(df_dev, "cuda", batch_size, bert_tokenizer, 256)
    df_test, dataloader_test = text_to_dataloader(df_test, "cuda", batch_size, bert_tokenizer, 256)

    test_f1_list = []
    test_acc_list = []

    for layers_count_i in [2,4,5,6,7,8,10,11]:

        save_path = f"Bert_base_frozen_pos_linaer_layers_{layers_count_i}_conf_mat.jpg"
        bert_config = transformers.BertConfig(num_hidden_layers=layers_count_i)
        model = BertProbeClassifer("cuda", 256, batch_size, "test_model", "bert-base-uncased", True, classes_count, bert_config)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), weight_decay=1e-3, lr=1e-4)

        model.fit(
            train_dataloader=dataloader_train,
            train_len=train_len,
            epochs=epochs,
            criterion=criterion,
            optimizer=optimizer,
            verbose=True,
            device="cuda",
            test_dataloader=dataloader_dev,
            test_len=test_len,
            save_checkpoints=False,
            model_save_threshold=0
        )

        # test

        label_list = list(set(df_test["label"]))
        label_list.sort()
        y_true, y_pred = calc_performance(model, dataloader_test)
        test_acc, test_f1 = plot_confusion_matrix(
            y_true,
            y_pred,
            normalize=True,
            label_list=label_list,
            savepath=save_path,
            visible=False
        )

    print("acc")
    print(test_acc_list)
    print("f1")
    print(test_f1_list)








