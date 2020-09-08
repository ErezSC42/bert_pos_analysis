import os
import tqdm
import torch
import datetime
import numpy as np
import transformers
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
            metric_func=check_accuracy_classification,
            save_checkpoints=False,
            model_save_threshold=0.85) -> dict:
        self.train()
        results = dict()
        results["train_acc"] = list()
        results["train_loss"] = list()
        if test_dataloader is not None:
            results["test_acc"] = list()
            results["test_loss"] = list()
        self.to(device)
        if verbose:
            print("statring training...")
        for epoch in tqdm.tqdm(range(epochs)):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(tqdm.tqdm(train_dataloader)):  # mini-batch
                self.train()
                inputs, mask, labels = data
                outputs = self(inputs, mask)
                real_labels = torch.max(labels, 1)[1] # TODO important to change like this for multiclass
                loss = criterion(outputs, real_labels) # TODO important to change like this for multiclass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # print statistics
                running_loss += loss.item()
            train_loss, train_acc = metric_func(train_dataloader, self, "train", train_len, verbose=verbose, criterion=criterion, use_masks=self.use_attn_masks, drop_grad=self._drop_grad)
            print(train_acc)
            results["train_acc"].append(train_acc)
            results["train_loss"].append(train_loss)
            if test_dataloader is not None:
                test_loss, test_acc = metric_func(test_dataloader, self, "test", test_len, verbose=verbose, criterion=criterion, use_masks=self.use_attn_masks, drop_grad=self._drop_grad)
                results["test_acc"].append(test_acc)
                results["test_loss"].append(test_loss)
                if save_checkpoints and model_save_threshold <= test_acc:
                    model_name = self.generate_model_save_name(test_acc)
                    model_path = os.path.join("models", model_name)
                    torch.save(self, model_path)
            if use_nni:
                nni.report_intermediate_result(test_acc)
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