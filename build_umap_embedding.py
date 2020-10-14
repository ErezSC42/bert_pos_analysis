import os
import umap
import tqdm
import pickle
import pandas as pd


#MODEL_LIST = ['bert', 'roberta']
#MODEL_LIST = ['roberta']
MODEL_LIST = ['bert']

if __name__ == '__main__':
    for model_name in MODEL_LIST:
        for i in tqdm.tqdm(range(1, 13)):
            bert_results_path = os.path.join("bert_embeddings", f"{model_name}_base_uncased_embedding_layer_{i}.pkl")
            query_df = pd.read_csv(bert_results_path)
            query_df.dropna(inplace=True)
            embedding_columns_list = [str(i) for i in list(range(768))]
            contextual_embedding_array = query_df[embedding_columns_list].to_numpy()
            query_df.drop(columns=embedding_columns_list, inplace=True)
            reducer = umap.UMAP()
            lower_dim_data = reducer.fit_transform(
                contextual_embedding_array,
            )
            file_path = os.path.join("umap_pickles", f"{model_name}_base_embedding_layer_{i}.pkl")
            saved_file = query_df, lower_dim_data
            with open(file_path, "wb") as fp:
                pickle.dump(saved_file, fp)