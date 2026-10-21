# Biological Query Classification

## Overview

This project evaluates the use of a lightweight large language model (LLM) for classifying biological information retrieval queries. Inspired by TourSynbio-Search, the experiment focuses on query classification rather than implementing the complete search framework.

The model used is **Qwen2.5-1.5B**, accessed locally through Ollama.

## Task Definition

Each query is classified into one of three categories:

- **PaperSearch:** Requests to find scientific papers or publications.
- **ProteinSearch:** Requests to retrieve protein sequences, structures, or database records.
- **GeneralQuestion:** Requests for biological explanations without retrieving external data.

## Experimental Setup

Three prompting configurations were evaluated:

1. Zero-shot prompting
2. Original few-shot prompting with six labeled examples
3. Revised few-shot prompting with clearer classification rules

The dataset contains 30 manually labeled queries, with 10 queries assigned to each category. The model predicts one label for each query, and accuracy is calculated by comparing predictions with the gold labels.

## Experimental Results

- Zero-shot: 96.67% accuracy (29/30 correct)
- Original few-shot: 33.33% accuracy (10/30 correct)
- Revised few-shot: 96.67% accuracy (29/30 correct)

The revised few-shot experiment used the same evaluation dataset after the original errors had been examined. Its result should therefore be interpreted as exploratory rather than as an independent test of generalization.

## Repository Contents

- `adapted_interface.py`: Lightweight interface adapted from TourSynbio to communicate with Ollama.
- `experiment.py`: Experiment script for the original prompting setup.
- `experiment3.py`: Experiment script for the revised few-shot prompt.
- `test.csv`: Labeled evaluation dataset containing 30 queries.
- `examples.csv`: Six labeled demonstrations used for few-shot prompting.
- `prompts_original.txt`: Original zero-shot and few-shot prompts.
- `prompt_revised.txt`: Revised few-shot prompt.
- `results_original.csv`: Results from the original experiments.
- `results_revised.csv`: Results from the revised few-shot experiment.

## Requirements

- Python 3
- Ollama
- Qwen2.5-1.5B (`qwen2.5:1.5b`)

## Running the Experiments

First, install Ollama and download the model:

```bash
ollama pull qwen2.5:1.5b
```

Ensure that Ollama is running and its local API is available. Then, from the project directory, run the appropriate experiment script:

```bash
python3 experiment.py
```

For the revised few-shot experiment:

```bash
python3 experiment3.py
```

The scripts may require adjustment depending on the local environment and the model/API configuration.

## Limitations

This study uses a small dataset of 30 queries and a single language model. It evaluates query classification only and does not implement the complete TourSynbio-Search framework or retrieve information from biological databases. The revised prompt was evaluated on the same dataset used to identify the original errors, so further testing on an independent dataset is needed.

## Reference

TourSynbio-Search: A Large Language Model Driven Agent Framework for Unified Search Method for Protein Engineering.

Original project: https://github.com/tsynbio/TourSynbio
