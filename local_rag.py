from transformers import pipeline
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
import numpy as np
from llama_cpp import Llama
from keybert import KeyBERT
from text_to_num import alpha2digit
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
pd.set_option("display.expand_frame_repr", False)



def markdown_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Reads a Markdown file and extracts sections and paragraphs into a Pandas DataFrame.
    """
    records = []
    current_section = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            # Skip empty lines
            if not stripped: continue
            # Check if line is a Markdown header (starts with one or more '#')
            if stripped.startswith('#'):
                # Extract the title by stripping leading '#' characters and whitespace
                current_section = re.sub(r'^#+\s*', '', stripped)
            else:
                # If we encounter text before any header, label it as 'Uncategorized'
                section_label = current_section if current_section else "Uncategorized"
                records.append({
                    "section": section_label,
                    "paragraph": stripped
                })
    return pd.DataFrame(records)



def mds2df(lst): 
    df = pd.DataFrame(columns=["title", "section", "paragraph"])
    for item in lst: 
        filename = f"./documents/{item}.md"
        df_i = markdown_to_dataframe(filename)
        df_i["title"] = item
        df = pd.concat([df, df_i], ignore_index=True)
    return df



def getSearchTerms(query, kw_model, ner_model): 
    Q = alpha2digit(query, "en").title()
    outputs = []

    # key words
    results = kw_model.extract_keywords(Q, keyphrase_ngram_range=(1, 3), top_n=10)
    for res in results: 
        if(res[1] > 0.3): outputs.append(res[0])

    # NER 
    results = ner_model(Q)
    for res in results: 
        outputs.append(res['word'])

    # numbers 
    numbers = re.findall(r"-?\d+(?:\.\d+)?", Q)
    outputs = outputs + numbers

    return outputs  


def count_substring(string, substring):
    """
    Count non-overlapping occurrences of substring in string,
    ignoring case.
    """
    string = alpha2digit(string, "en")
    return string.lower().count(substring.lower())

def getKW_score(df, KWs):
    df2 = df.copy(deep=True)
    for kw in KWs: 
        df2[kw] = 0
    for i in range(len(df2)): 
        text = df2.at[i, 'paragraph']
        for kw in KWs: 
            df2.at[i, kw] = count_substring(text, kw)

    totals = df2.sum().to_list()[3:]
    totals = [1 if x == 0 else x for x in totals]
    df2['KWscore'] = 0.0

    for i in range(len(df2)): 
        row = df2.iloc[i]
        counts = row.to_list()[3:-1]
        percentages = np.divide(counts, totals)
        score = max(percentages)
        df2.at[i, 'KWscore'] = score
    return df2[['title', 'section', 'paragraph', 'KWscore']]


def getScores(paraDF, query, text_encoder): 
    df = paraDF.copy(deep=True)
    df = df.reset_index(drop=True)
    df["semanticScore"] = 0.0
    query_enc = text_encoder.encode("Represent this sentence for searching relevant passages:"+query)
    for i in range(len(df)): 
        text = df.at[i, "paragraph"]
        sentences = sent_tokenize(text)
        txt_enc = text_encoder.encode(sentences)
        scores = util.dot_score(query_enc, txt_enc)[0].cpu().tolist()
        df.at[i, "semanticScore"] = max(scores)
    return df

def getAnswer(query, title, section, paragraph, llm, max_tokens):
    model_input = f"Given this paragraph context, tell me if this query can be answered. If it can be answered, return the answer. If it cannot, return 'Not enough context.' \n\nQuery: {query}\n\nTitle: {title}\nSection: {section}\nParagraph: {paragraph}"
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": model_input}
        ],
        temperature=0.2,
        max_tokens=max_tokens
    )
    return response["choices"][0]["message"]["content"]

def query_documents(df, llm, K=5, max_tokens=2048): 
    df2 = df.copy(deep=True).reset_index(drop=True)
    answer = ""
    citation = None
    for i in range(min(len(df2), K)): 
        row = df2.iloc[i]
        title, section, paragraph = row[["title", "section", "paragraph"]].to_list()
        resp = getAnswer(query, title, section, paragraph, llm, max_tokens)
        if("</think>" in resp):
            resp = resp.split("</think>")[-1]

        if("not enough context" not in resp.lower()): 
            answer = resp
            citation = (title, section)
            break

    if(answer == ""):   return "Answer to query not found in documents.", None
    else:               return answer, citation


def queryDB(query, documents, llm, kw_model, ner_model, text_encoder, verbose=True): 
    K_kw = 50
    K_sm = 20 
    K_qu = 10

    # Get Documents
    df = mds2df(documents)
    if(verbose): print(f"\nFetched text from {len(documents)} documents:\n\t{documents}")

    # Filtering by Words
    search_terms = getSearchTerms(query, kw_model=kw_model, ner_model=ner_model)
    if(verbose): print(f"\nUsing the following key words for filtering:\n\t{search_terms}")
    df = getKW_score(df, search_terms)
    df = df.sort_values(by="KWscore", ascending=False).reset_index(drop=True)
    df = df[:K_kw]

    # Filtering by Semantics
    df = getScores(df, query, text_encoder)
    df = df.sort_values(by="semanticScore", ascending=False).reset_index(drop=True)
    df = df[:K_sm]

    # Sorting by Geometric Mean
    df["geometricMean"] = np.sqrt(df["KWscore"] * df["semanticScore"])
    df = df.sort_values(by="geometricMean", ascending=False).reset_index(drop=True)
    df = df[:K_qu]

    # Querying Remaining Paragraphs
    answer = query_documents(df, llm, K=K_qu)
    return answer



if __name__ == "__main__": 

    # Models 
    llm = Llama(model_path="models/Phi-4-mini-reasoning-Q4_K_M.gguf", n_ctx=4096, n_gpu_layers=-1, verbose=False) # -1 offloads all layers to GPU
    kw_model = KeyBERT("all-MiniLM-L6-v2")
    text_encoder = SentenceTransformer("./models/bge-base-en-v1.5")
    ner_model = pipeline("token-classification", model="./models/NER", aggregation_strategy="simple")

    # Inputs
    query = "Who was the 46th president of the United States?"
    documents = ["Ted_Cruz", "Tim_Kaine", "Rand_Paul"]

    # Get Answer 
    answer, citation = queryDB(query, documents, 
                     llm=llm, 
                     kw_model=kw_model, 
                     ner_model=ner_model, 
                     text_encoder=text_encoder, 
                     verbose=False)
    print(f"Model output: {answer}")
    print(f"Citation: {citation}")

    # Cleanup 
    llm.close()