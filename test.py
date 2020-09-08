import os
import torch
import pandas as pd
import transformers
from utils import text_to_dataloader, tokenize_word


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

    bert_tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    text_to_dataloader(df_train, "cuda", 32, bert_tokenizer, 256)



