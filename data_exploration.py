#!/usr/bin/env python
# coding: utf-8
import os
import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from utils import check_accuracy_classification
import transformers
from torch.optim import Adam
from models import BertProbeClassifer
from utils import text_to_dataloader, tokenize_word
from bert_embedding import BertEmbeddingExtractor



if __name__ == '__main__':



    train_path = os.path.join("data","en_partut-ud-train.conllu")
    dev_path = os.path.join("data","en_partut-ud-dev.conllu")
    test_path = os.path.join("data","en_partut-ud-test.conllu")


    # In[4]:


    HEADER_CONST = "# sent_id = "
    TEXT_CONST = "# text = "
    STOP_CONST = "\n"
    WORD_OFFSET = 1
    LABEL_OFFSET = 3


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
                    df = pd.concat([df,cur_df])
            return df



    # In[5]:


    df_train = txt_to_dataframe(train_path)


    # In[6]:


    TYPES = [
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
        "_"
    ]


    # In[7]:


    file_name = 'tex_artifacts/label_dist_train.tex'
    SORT_COL = "Count"

    with open(file_name,'w') as tf:
        display_df = df_train["label"].value_counts().rename_axis("Type").to_frame("Count").reset_index()
        #display_df.index = TYPES
        display_df.sort_values(by=SORT_COL, inplace=True, ascending=False)
        latex_data = display_df.to_latex(index=False)
        tf.write(latex_data)

    # In[11]:


    bert_tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


    # In[12]:

    df_train, dataloader_train = text_to_dataloader(df_train, "cuda", 32, bert_tokenizer, 256)

    # In[14]:


    import warnings
    warnings.filterwarnings(action="ignore")


    # extract bert base embedding
    for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
        bex = BertEmbeddingExtractor(i, "bert-base-uncased")
        embedding_df = bex.extract_embedding(dataloader_train, "sum")

        break
        save_path = os.path.join("bert_embeddings", f"bert_base_embedding_layer_{i}")
        embedding_df.to_csv(save_path)


    # In[ ]:


    np.vstack(query_df["embedding"].apply(lambda x: np.array(list(x)))).shape


    # In[ ]:


    import umap


    chosen_bert_layer = 12
    bert_results_path = os.path.join("bert_embeddings", f"bert_base_embedding_layer_{chosen_bert_layer}")

    query_df = pd.read_csv(
        bert_results_path,
        dtype={"embedding": np.array}
    )

    contextual_embedding_array = np.vstack(query_df["embedding"].values)

    reducer = umap.UMAP()
    lower_dim_data = reducer.fit_transform(
        contextual_embedding_array,
        y=query_df["label_idx"].tolist()
    )


    # In[ ]:


    import matplotlib.pyplot as plt
    from pylab import cm
    import mplcursors

    get_ipython().run_line_magic('matplotlib', 'qt')

    word_list = list(query_df["word"].tolist())
    all_labels = query_df["label"].tolist()
    labels = list(set(all_labels))
    labels.sort()
    n_colors = len(labels)


    #create new colormap
    cmap = cm.get_cmap('tab20', n_colors)


    print(n_colors)


    fig, ax = plt.subplots(figsize=(10,10))

    sc = plt.scatter(
        lower_dim_data[:,0],
        lower_dim_data[:,1],
        c=query_df["label_idx"].tolist(),
        cmap=cmap,
        s=1
    )

    # cursor
    crs = mplcursors.cursor(ax,hover=True)
    crs.connect(
        "add",
        lambda sel: sel.annotation.set_text(
            f"{word_list[sel.target.index]}\n{all_labels[sel.target.index]}"
        ))


    # colorbar
    c_ticks = np.arange(n_colors) * (n_colors / (n_colors + 1)) + (2 / n_colors)
    cbar = plt.colorbar(sc, ticks=c_ticks)
    #cbar = plt.colorbar()

    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(labels, ha="right")
    cbar.ax.yaxis.set_tick_params(pad=40)
    plt.show()


    # In[ ]:


    labels


    # In[ ]:


    set(display_labels)


    # In[ ]:


    display_labels


    # In[ ]:




