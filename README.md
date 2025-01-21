# MARS - Predicting Correctness from Model Internals

## Next steps

Short-term steps:

- Perform correctness experiment with gsm8k data with and without CoT
- Perform analysis to find best layer (Anton?)
- For countries dataset, compute activations after response and compare accuracy
- Check probabilites of the IDK token for all questions

Potential future steps:
- If we see a difference between correctness accuracies pre and post response, check the tokens in between. In other works, check when the model "becomes aware" of what it knows. 
- Reversal curse paper: see where each direction (A -> B or B -> A) goes in our plot.
- For the countries dataset, get baseline for correctness by prompting the model about its confidence. 

## Context of this work

The goal of this work is to understand how much self-aware models are of their ignorance by analysing its internals. It is inspired by the following two papers: 

- **Truth is Universal: Robust Detection of Lies in LLMs** (https://arxiv.org/abs/2407.12831): This paper finds "the direction of truth" to predict whether a statement is true from internals. The tecniques used here are simple and explained.
- **Language Models (Mostly) Know What They Know** (https://arxiv.org/abs/2207.05221): This paper predicts correctness from internals, but they don't share their techniques. 

We have tried to reproduce the results from the second paper using similar techniques from the first paper. 

## Datasets

### Cities dataset

For our experiments, we needed a dataset such that

1) The questions are simple enough so that the internals do not become too complicated.
2) The questions are hard enough for the model so that it gives wrong answers to a substantaial amount of the questions.

For that, we took a dataset of all the cities in the world with over 1000 people and asked the model to predict the country where the city is located in the following format.

Cities dataset: https://public.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000/table/?disjunctive.cou_name_en&sort=name

*In which country is the city of Barcelona located?*

We find that randomlly sampling from this dataset, Llama3 8B would be correct in about 75% of cases, which gives enough wrong answers. We also find that there are 3 kind of solutions that we could get from the model:

1) The model gives the right answer
2) The model gives a wrong answer
3) The model says "I don't know"

A first PCA of the activations in layer 12 of Llama3 8B already shows some interesting results. 

(add plots)

#### Potential similar datasets for the future

- Birth years of famous people

### GSM8K

We have run the activations and generations for 200 analysis but not yet performed the correctness analysis

## Using the repository

### Downloading HF resources

> Remember to store on pod's persistent volume mounted at `/workspace`.

- Download `huggingface-cli`: `pip install -U "huggingface_hub[cli]"`.
- Login with HF token before downloading for gated resources (`huggingface-cli login`).
- Download model: `huggingface-cli download --local-dir /workspace/models/<local_model_id> <model_id_hf>`.

Resources directory structure:

- Models: `/workspace/models`
- Datasets: `/workspace/datasets`

### Experiments

Sample number: 1000 samples?

1. PCA / tSNE on activations at ${l_i, ..., l_n}$, where $l_i \in$ residual stream transformer layers, on settings:
    - Prompt only
    - Full generation (prompt + generated answer)
    - (opt) CoT setting: (prompt + CoT, but no answer)
        - Potential limitation: how to detect end of CoT / start of answer in model generations
2. Visualize plots in search of clusters across $n$ layers with clear separation among correct / incorrect generations. With the identified layer, train a linear classifier for the previously defined settings.
    - Single parameter: answer correctness

Evaluation: compare model generations against ground truth to obtain labels. 
