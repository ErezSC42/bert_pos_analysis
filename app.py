import os
import umap
import altair as alt
import streamlit as st
import pandas as pd


@st.cache
def load_data(chosen_bert_layer):
    bert_results_path = os.path.join("bert_embeddings", f"bert_base_embedding_layer_{chosen_bert_layer}")
    query_df = pd.read_csv(bert_results_path)
    query_df.dropna(inplace=True)
    embedding_columns_list = [str(i) for i in list(range(768))]
    contextual_embedding_array = query_df[embedding_columns_list].to_numpy()
    query_df.drop(columns=embedding_columns_list, inplace=True)
    reducer = umap.UMAP()
    lower_dim_data = reducer.fit_transform(
        contextual_embedding_array,
    )
    del contextual_embedding_array
    return query_df, lower_dim_data


st.title("BERT")
st.write("Choose BERT layer")

chosen_bert_layer = st.slider(
    min_value=1,
    max_value=12,
    label="Layer",
    step=1,
    value=4
)

query_df, lower_dim_data = load_data(chosen_bert_layer)

query_text = st.text_input(
    label="Select Word",
    max_chars=15,
    value="",
)


if query_text == "":
    viz_df = query_df
    matching_idx = len(query_df) * [True]
else:
    matching_idx = query_df["word"] == query_text
    viz_df = query_df[matching_idx]
st.write(f"Results: {len(viz_df)}")

viz_df["X"] = lower_dim_data[matching_idx,0]
viz_df["Y"] = lower_dim_data[matching_idx,1]
viz_df.dropna(inplace=True)


chart = alt.Chart(viz_df).mark_point(size=10).encode(
    x="X",
    y="Y",
    color="label",
    tooltip=['word', 'label', 'text'],
).interactive()

st.altair_chart(chart, use_container_width=True)
st.write(viz_df.head(10))
