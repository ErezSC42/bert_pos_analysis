import os
import random
import warnings
import pickle
import altair as alt
import streamlit as st
import streamlit_theme as stt

stt.set_theme({
    'primary':'#00cc99',
})

SAMPLES_TO_DISPALY = 20
X_LIM_MIN = -20
X_LIM_MAX = 20
Y_LIM_MIN = -20
Y_LIM_MAX = 22
seed = 42
warnings.filterwarnings(action="ignore")

LABELS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB"]

#@st.cache
def load_data():
    temp_bert = {}
    temp_roberta = {}
    for i in range(1,13):
        bert_results_path = os.path.join("umap_pickles", f"bert_base_embedding_layer_{i}.pkl")
        roberta_results_path = os.path.join("umap_pickles", f"roberta_base_embedding_layer_{i}.pkl")
        with open(bert_results_path, "rb") as fp:
            temp_bert[i] = pickle.load(fp)
        with open(roberta_results_path, "rb") as fp:
            temp_roberta[i] = pickle.load(fp)
    return temp_bert, temp_roberta

bert_embeddings_dataframes, roberta_embeddings_dataframes = load_data()

st.title("BERT Embedding Explorer")
st.write("Choose BERT layer")

chosen_bert_layer = st.slider(
    min_value=1,
    max_value=12,
    label="Layer",
    step=1,
    value=4
)

chosen_model = st.radio(
    "Choose Model:",
    ("bert-base", "roberta-base")
)

if chosen_model == "bert-base":
    query_df, lower_dim_data = bert_embeddings_dataframes[chosen_bert_layer]
elif chosen_model == "roberta-base":
    query_df, lower_dim_data = roberta_embeddings_dataframes[chosen_bert_layer]

selected_input_type = st.radio(
    "Select Input Mode",
    ("row", "word")
)

input_display = "Word" if selected_input_type == "word" else "Row Index"
boolean_text_display = False if selected_input_type == "word" else True

if selected_input_type == "row":
    query_text = st.number_input(
        label=f"Select {input_display}",
        value=seed
    )
else:
    query_text = st.text_input(
        label=f"Select {input_display}",
        max_chars=15,
        value="",
    )


viz_df = query_df
if query_text == "":
    matching_idx = len(query_df) * [True]
else:
    if selected_input_type == "word":
        matching_idx = query_df["word"] == query_text
    else:  #row
        row_text = query_df["text"].loc[query_text]
        matching_idx = query_df["text"] == row_text
    viz_df = query_df[matching_idx]
    if boolean_text_display:
        text_display = st.write(f"{row_text}")
    st.write(f"Results: {len(viz_df)}")
viz_df["X"] = lower_dim_data[matching_idx,0]
viz_df["Y"] = lower_dim_data[matching_idx,1]
viz_df.dropna(inplace=True)


chart = alt.Chart(viz_df).mark_point(size=50, filled=True).encode(
    alt.X("X", scale=alt.Scale(domain=(X_LIM_MIN, X_LIM_MAX))),
    alt.Y("Y", scale=alt.Scale(domain=(Y_LIM_MIN, Y_LIM_MAX))),
    color="label",
    tooltip=['word', 'label', 'text']
).interactive()


st.altair_chart(chart, use_container_width=True)
st.write(viz_df.head(SAMPLES_TO_DISPALY))
