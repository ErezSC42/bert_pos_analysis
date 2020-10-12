import os
import umap
import tqdm
import pickle
import pandas as pd


if __name__ == '__main__':
    for i in tqdm.tqdm(range(1, 13)):
        bert_results_path = os.path.join("bert_embeddings", f"bert_base_embedding_layer_{i}")
        query_df = pd.read_csv(bert_results_path)
        query_df.dropna(inplace=True)
        embedding_columns_list = [str(i) for i in list(range(768))]
        contextual_embedding_array = query_df[embedding_columns_list].to_numpy()
        query_df.drop(columns=embedding_columns_list, inplace=True)
        reducer = umap.UMAP()
        lower_dim_data = reducer.fit_transform(
            contextual_embedding_array,
        )
        file_path = os.path.join("umap_pickles", f"bert_base_embedding_layer_{i}")
        saved_file = query_df, lower_dim_data
        with open(file_path, "wb") as fp:
            pickle.dump(saved_file, fp)