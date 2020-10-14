import os
import json
import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import check_accuracy_classification
import transformers
from torch.optim import Adam
from models import BertProbeClassifer
from utils import text_to_dataloader, tokenize_word

CPU = "cpu"


class BertEmbeddingExtractor():
    def __init__(
            self,
            bert_layer: int,
            bert_base_model : str = "bert-base-uncased"):

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # bert config
        self.bert_base_model = bert_base_model
        # bert
        self.bert_layer = bert_layer

    def extract_embedding(self, dataloader : DataLoader, agg_func: str) -> pd.DataFrame:
        '''
        :param dataloader: text_to_dataloader(df_train, "cuda", 32, bert_tokenizer, 256)
        :return:
        '''
        self.bert_model.to(self.device)
        labels = []
        orig_words_list = []
        contextual_embeddings = []

        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                batch_text, batch_mask, target_word_mask, batch_labels = batch
                batch_contextual_embeddings, _ = self.bert_model(batch_text, batch_mask)

                current_batch_size = batch_contextual_embeddings.shape[0]

                # get original word as sting
                batch_target_word_ids = torch.mul(target_word_mask, batch_text)
                batch_list_target_word_ids = [list(filter(lambda num: num != 0, l)) for l in
                                              batch_target_word_ids.tolist()]
                batch_original_words_list = [
                    self.bert_tokenizer.convert_tokens_to_string(self.bert_tokenizer.convert_ids_to_tokens(l)) for l in
                    batch_list_target_word_ids]
                orig_words_list = orig_words_list + batch_original_words_list

                # get embedding vector
                bool_mask = target_word_mask.cpu().type(torch.BoolTensor)
                for batch_idx in range(current_batch_size):
                    q = batch_contextual_embeddings[batch_idx, bool_mask[batch_idx, :], :].cpu()
                    if q.shape[0] > 1:
                        if agg_func == "sum":
                            q = torch.sum(q, axis=0)  # aggregation of tokens vectors of same word
                        elif agg_func == "mean":
                            q = torch.mean(q, axis=0)
                    contextual_embeddings.append(q.numpy().squeeze())

                # get pos_label
                labels = labels + batch_labels.cpu().tolist()

        # free gpu memory
        self.bert_model.to(CPU)

        embedding_df = pd.DataFrame({
            "word": orig_words_list,
            "label_idx": labels
        })
        vector_df = pd.DataFrame.from_records(contextual_embeddings)
        embedding_df = pd.concat([embedding_df, vector_df], axis=1)
        embedding_df[embedding_df["word"] != ""]

        with open("pos_to_label.json", "rb") as fp:
            pos_to_label_dict = json.load(fp)
            inv_map = {v: k for k, v in pos_to_label_dict.items()}
            embedding_df["label"] = embedding_df["label_idx"].map(inv_map)
        return embedding_df


class BertEmbeddingExtractorVanilla(BertEmbeddingExtractor):
    def __init__(
            self,
            bert_layer: int,
            bert_base_model : str = "bert-base-uncased"):
        super(BertEmbeddingExtractorVanilla, self).__init__(
            bert_layer=bert_layer,
            bert_base_model=bert_base_model
        )
        self.bert_config = transformers.AutoConfig.from_pretrained(
            bert_base_model,
            num_hidden_layers=bert_layer
        )
        self.bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.bert_base_model
        )
        self.bert_model = transformers.AutoModel.from_pretrained(
            config=self.bert_config,
            pretrained_model_name_or_path=bert_base_model
        )


class BertEmbeddingExtractorRandom(BertEmbeddingExtractor):
    def __init__(
            self,
            bert_layer: int,
            bert_base_model : str = "bert-base-uncased"):
        super(BertEmbeddingExtractorRandom, self).__init__(
            bert_layer=bert_layer,
            bert_base_model=bert_base_model
        )
        self.bert_config = transformers.BertConfig(num_hidden_layers=bert_layer)
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.bert_base_model
        )
        self.bert_model = transformers.BertModel(
            config=self.bert_config,
        )

