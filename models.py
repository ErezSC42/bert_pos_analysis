import os
import tqdm
import torch
import datetime
import numpy as np
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import calc_performance,calc_classification_metrics

import warnings
warnings.filterwarnings(action="ignore")


class BertWrapperModel(nn.Module):
    '''
         this class implements a predict method for
         pytorch based models for text inference.
        :param input - string or pd.Series
    '''
    def __init__(
            self,
            device : torch.device,
            max_sequence_len : int,
            inference_batch_size : int,
            model_name : str,
            bert_pretrained_model: str,
            freeze_bert: bool,
            bert_config: transformers.BertConfig = None):
        nn.Module.__init__(self)
        self.device = device
        self.max_sequence_len = max_sequence_len
        self.inference_batch_size = inference_batch_size
        self.model_name = model_name
        self.freeze_bert = freeze_bert
        self.bert_config = bert_config

        self.tokenizer = transformers.BertTokenizer.from_pretrained(bert_pretrained_model)
        self.bert = transformers.BertModel.from_pretrained(bert_pretrained_model, config=bert_config)

        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def generate_model_save_name(self, test_acc):
        return f"{self.__class__.__name__}_{self._lang}_{self._task}_{self._model_name}_{test_acc:.3}_{datetime.datetime.today().day}_{datetime.datetime.today().month}.pth".replace("-", "_").replace("/", "_")

    def fit(self,
            train_dataloader: DataLoader,
            train_len: int,
            epochs: int,
            criterion: nn.CrossEntropyLoss,
            optimizer: torch.optim.Adam,
            verbose=True,
            device="cuda",
            test_dataloader=None,
            test_len=None,
            use_nni=False,
            save_checkpoints=False,
            model_save_threshold=0.85) -> dict:
        self.train()
        results = dict()
        results["train_acc"] = list()
        results["train_loss"] = list()
        results["train_precision"] = list()
        results["train_recall"] = list()
        results["train_f1"] = list()
        if test_dataloader is not None:
            results["test_acc"] = list()
            results["test_loss"] = list()
            results["test_precision"] = list()
            results["test_recall"] = list()
            results["test_f1"] = list()
        self.to(device)
        if verbose:
            print("statring training...")
        for epoch in tqdm.tqdm(range(epochs)):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(tqdm.tqdm(train_dataloader)):  # mini-batch
                self.train()
                inputs, mask, target_mask,labels = data
                outputs = self(inputs, mask, target_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # print statistics
                #running_loss += loss.item()
            y_real, y_pred = calc_performance(self, train_dataloader)
            acc, precision, recall, f1 = calc_classification_metrics(y_true=y_real, y_pred=y_pred)
            if verbose:
                print(f'\tEp #{epoch} | Train. Loss: {loss:.3f} | Acc: {acc * 100:.2f}% | Precision: {precision * 100:.2f}% | Recall: {recall * 100:.2f}% | F1: {f1 * 100:.2f}%')
            results["train_acc"].append(acc)
            results["train_loss"].append(loss)
            results["train_precision"].append(precision)
            results["train_recall"].append(recall)
            results["train_f1"].append(f1)
            if test_dataloader is not None:
                y_real, y_pred = calc_performance(self, test_dataloader)
                acc, precision, recall, f1 = calc_classification_metrics(y_true=y_real, y_pred=y_pred)
                if verbose:
                    print(
                        f'\tEp #{epoch} | Val. Acc: {acc * 100:.2f}% | Precision: {precision * 100:.2f}% | Recall: {recall * 100:.2f}% | F1: {f1 * 100:.2f}%')
                results["train_acc"].append(acc)
                results["train_loss"].append(loss)
                results["train_precision"].append(precision)
                results["train_recall"].append(recall)
                results["train_f1"].append(f1)
                if save_checkpoints and model_save_threshold <= acc:
                    model_name = self.generate_model_save_name(acc)
                    model_path = os.path.join("models", model_name)
                    torch.save(self, model_path)
            if use_nni:
                nni.report_intermediate_result(acc)
        if verbose:
            print('Finished Training')
        return results

    def predict_proba(self, sentences, use_batch=True):
        '''
        :param sentences: str, list(str), pd.Series(str) - collection of strings to inference on
        :param use_batch: use batch prediction for performance. valid only when sentences != str
        :return: returns np.array([sentences_len, classes_len]), where axis=1 is the probability of each class
        '''
        self.eval()
        self.to(self.device)
        inference_dataloader = text_to_dataloader(
            sentences,
            self.device,
            self._inference_batch_size,
            self.tokenizer,
            self.max_sequence_len
        )
        predictions = []
        with torch.no_grad():
            for batch_data in inference_dataloader:
                batch_text_ids, batch_attn_mask = batch_data
                batch_outputs, attn = self(batch_text_ids, batch_attn_mask)
                probas = batch_outputs.cpu().detach().numpy()
                predictions.append(probas)
        predicaions_proba = np.exp(np.vstack(predictions))  # [sentences, class_num] 2d-array
        return predicaions_proba

    def predict(self, sentences, use_batch=True):
        probas = self.predict_proba(sentences, use_batch)
        return np.argmax(probas, axis=1)

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BertProbeClassifer(BertWrapperModel):
    def __init__(
            self,
            device : torch.device,
            max_sequence_len : int,
            inference_batch_size : int,
            model_name : str,
            bert_pretrained_model: str,
            freeze_bert: bool,
            class_count: int,
            bert_config: transformers.BertConfig = None):
        super(BertProbeClassifer, self).__init__(
            device=device,
            max_sequence_len=max_sequence_len,
            inference_batch_size=inference_batch_size,
            model_name=model_name,
            bert_pretrained_model=bert_pretrained_model,
            freeze_bert=freeze_bert,
            bert_config=bert_config,
        )
        self.class_count = class_count
        self.linear = nn.Linear(in_features=768, out_features=class_count)

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    @staticmethod
    def entity_sum(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attn_mask, target_word_mask):
        embeddings, _ = self.bert(input_ids, attn_mask)
        target_word_embedding = self.entity_sum(embeddings, target_word_mask)
        return F.log_softmax(self.linear(target_word_embedding), dim=1)



