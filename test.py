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

    #train_len = len(df_train)
    train_len = len(df_train)
    test_len = len(df_dev)

    bert_tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    batch_size = 64

    df_train = df_train.head(train_len)
    df_dev = df_dev.head(test_len)

    dataloader_train = text_to_dataloader(df_train, "cuda", batch_size, bert_tokenizer, 256)
    dataloader_dev = text_to_dataloader(df_dev, "cuda", batch_size, bert_tokenizer, 256)
    dataloader_test = text_to_dataloader(df_test, "cuda", batch_size, bert_tokenizer, 256)

    model = BertProbeClassifer("cuda", 256, batch_size, "test_model", "bert-base-uncased", True, 18, None)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), weight_decay=1e-3, lr=1e-4)

    model.fit(
        train_dataloader=dataloader_train,
        train_len=train_len,
        epochs=5,
        criterion=criterion,
        optimizer=optimizer,
        verbose=True,
        device="cuda",
        test_dataloader=dataloader_dev,
        test_len=test_len,
        metric_func=check_accuracy_classification,
        save_checkpoints=False,
        model_save_threshold=0
    )

    # test
    with open("pos_to_label.json", "rb") as fp:
        label_list = list(json.load(fp).keys())
        y_true, y_pred = calc_performance(model, dataloader_test)
        plot_confusion_matrix(y_true, y_pred, normalize=True, label_list=label_list)



