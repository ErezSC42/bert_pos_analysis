import os
import nni
import torch
import transformers
import torch.nn as nn
from torch.optim import Adam
from argparse import Namespace
from arguments import get_params
from models import BertProbeClassifer
from utils import check_accuracy_classification, txt_to_dataframe
from utils import text_to_dataloader, plot_confusion_matrix, calc_performance


#MODEL_NAME = "bert-base-uncased"
MODEL_NAME = "roberta-base"
MATRICES_SAVE_PATH = os.path.join("confusion_matrices")


def main(args):
    # set seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = os.path.join("data", "en_partut-ud-train.conllu")
    dev_path = os.path.join("data", "en_partut-ud-dev.conllu")
    test_path = os.path.join("data", "en_partut-ud-test.conllu")

    df_train = txt_to_dataframe(train_path)
    df_dev = txt_to_dataframe(dev_path)
    df_test = txt_to_dataframe(test_path)

    train_len = len(df_train)
    test_len = len(df_dev)

    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    # getting arguments
    batch_size = args.batch_size
    epochs = args.epochs
    classes_count = args.class_count
    num_hidden_layers = args.hidden_layers_count
    max_sequence_len = args.max_sequence_len
    freeze_bert = args.freeze_bert
    random_weights = args.random_weights

    df_train = df_train.head(train_len)
    df_dev = df_dev.head(test_len)

    df_train, dataloader_train = text_to_dataloader(df_train, "cuda", batch_size, bert_tokenizer, max_sequence_len)
    df_dev, dataloader_dev = text_to_dataloader(df_dev, "cuda", batch_size, bert_tokenizer, max_sequence_len)
    df_test, dataloader_test = text_to_dataloader(df_test, "cuda", batch_size, bert_tokenizer, max_sequence_len)

    model_name = "test_model"

    if random_weights:
        save_path = os.path.join("confusion_matrices",
                                 f"{MODEL_NAME}_random_frozen_pos_linaer_layers={num_hidden_layers}_seed={seed}_conf_mat.jpg")
    else:
        save_path = os.path.join("confusion_matrices",
                                 f"{MODEL_NAME}_frozen_pos_linaer_layers={num_hidden_layers}_seed={seed}_conf_mat.jpg")
    bert_config = transformers.AutoConfig.from_pretrained(
        MODEL_NAME,
        num_hidden_layers=num_hidden_layers
    )
    model = BertProbeClassifer(
        device=device,
        max_sequence_len=max_sequence_len,
        inference_batch_size=batch_size,
        model_name=model_name,
        bert_pretrained_model=MODEL_NAME,
        freeze_bert=freeze_bert,
        class_count=classes_count,
        bert_config=bert_config,
        random_weights=random_weights
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    model.fit(
        train_dataloader=dataloader_train,
        train_len=train_len,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        verbose=True,
        device=device,
        use_nni=args.use_nni,
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
    print(test_f1)
    print("f1")
    print(test_f1)

    if args.use_nni:
        nni.report_final_result(test_acc)


if __name__ == '__main__':
    try:
        # get parameters from tuner
        namespace_params = get_params()
        if namespace_params.use_nni:
            print("nni activated.")
            tuner_params = nni.get_next_parameter()
            params = vars(namespace_params)
            print("TUNER PARAMS: " + str(tuner_params))
            print("params:" + str(params))
            params.update(tuner_params)
            namespace_params = Namespace(**params)
        main(namespace_params)
    except Exception as ex:
        torch.cuda.empty_cache()
        raise






