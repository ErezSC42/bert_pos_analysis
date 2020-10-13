import os
import transformers
import pandas as pd
from utils import text_to_dataloader
from bert_embedding import BertEmbeddingExtractorVanilla

HEADER_CONST = "# sent_id = "
TEXT_CONST = "# text = "
STOP_CONST = "\n"
WORD_OFFSET = 1
LABEL_OFFSET = 3
NUM_OFFSET = 0


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
                num_list = []
                text = line.split(TEXT_CONST)[1]
                # this is a new text, need to parse all the words in it
            elif line is not STOP_CONST and HEADER_CONST not in line:
                temp_list = line.split("\t")
                num_list.append(temp_list[NUM_OFFSET])
                words_list.append(temp_list[WORD_OFFSET])
                labels_list.append(temp_list[LABEL_OFFSET])
            if line == STOP_CONST:
                # this is the end of the text, adding to df
                cur_df = pd.DataFrame(
                    {
                        "text": len(words_list) * [text],
                        "word": words_list,
                        "word_offset": num_list,
                        "label": labels_list,
                        "word_count" : len(words_list)
                    }
                )
                df = pd.concat([df, cur_df])
        return df


if __name__ == '__main__':

    #MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = "roberta-base"

    train_path = os.path.join("data", "en_partut-ud-train.conllu")
    df_train = txt_to_dataframe(train_path)
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    df_train, dataloader_train = text_to_dataloader(df_train.head(), "cuda", 32, bert_tokenizer, 256)
    print()

    bex = BertEmbeddingExtractorVanilla(1, MODEL_NAME)
    embedding_df = bex.extract_embedding(dataloader_train, "sum")
    print()

