# Local-RAG
A fully local RAG model for answering questions with answers found in text data. 

<img width="668" height="476" alt="RAG_Diagram" src="https://github.com/user-attachments/assets/a291b09e-df19-4c93-9eeb-9d627f217f48" />

RAG Process: 
1) Collect list of paragraphs from requested documents
2) Filter out paragraphs that do not include enough keywords from the query
3) Filter out paragraphs that are not semantically aligned with the query
4) Use an LLM to iterate over the remaining paragraphs. If the answer is found, return the answer. If not, ignore the paragraph.

Notes:
- This process will not return an answer that is not backed by piece of context. If all the paragraphs are iterated over and none contain the answer to the query, the model will return nothing instead of relying on training data to give its best guess. 
- This process can only return the first found answer which is a tradeoff made for more speed. If there are 2 answers found in different contexts, only the first will be returned. This is done so that not every paragraph is iterated over if an answer is found in the first.
- This process does not have multi-step reasoning but such a thing can be easily implemented in the future by calling this process for each sub-question in a larger question. 

## Loading Wiki Pages 
Text from Wikipedia pages can be added to a folder named `documents` by running the following script: 

```python load_wiki_pages.py```

By default, this program will create a `.md` file for every current US Senator. 

The format of the `.md` files is as follows: 

```
Title.md:
## Title 
# Section
paragraph 
paragraph 
...
# Section: Subsection
...
```

## Loading Models

This local RAG model depends on 4 local models which should be placed in a folder named `models`. Models 2-4 below can be downloaded using the script `./models/download_models.ipynb`

1. Any LLM represented by a `.gguf` file. This can be downloaded from `LMStudio`. 
2. The `bge-base-en-v1.5` folder
3. The `multi-qa-mpnet-base-dot-v1` folder
4. The `NER` folder

## Querying RAG Model

The documents can be searched for an answer to a query by setting the `query` variable and the `documents` variable in the file `local_rag.py` and running the Python script. Make sure to also define the `model_name` variable to the `.gguf` file name in your `./models` folder. 

## Examples

The following examples were generated using `qwen3-vl-4b` as the backbone LLM. The process can tolerate reasoning models but they are much slower and do not exhibit noticeable improvements in performance because the task is very simple.

```
Query: "Where is Tim Kaine a senator?"
Documents: "Ted_Cruz", "Tim_Kaine", "Rand_Paul"

Model output: 
Tim Kaine is a senator from Virginia.
Citation: ('Tim_Kaine', 'Political positions: Environment, energy, and climate change')
```

```
Query: "Which university did Ted Cruz attend?"
Documents: "Ted_Cruz", "Tim_Kaine", "Rand_Paul"

Model output: 
Ted Cruz attended Princeton University for his undergraduate studies and later Harvard LawSchool for his law degree.
Citation: ('Ted_Cruz', 'Early life and family: Education')
```

```
Query: "Who is a Senator for Kentucky?"
Documents: "Ted_Cruz", "Tim_Kaine", "Rand_Paul"

Model output: 
Rand Paul is a Senator for Kentucky.
Citation: ('Rand_Paul', 'Election to U.S. Senate: Primary campaign')
```

`
