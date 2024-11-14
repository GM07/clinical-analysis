# Ontology-Constrained Clinical Text Summarization
This repository contains the implementation of my master's thesis research at Polytechnique Montreal, which resulted in the paper "Ontology-Constrained Generation of Domain-Specific Clinical Summaries" published at EKAW 2024.

# Overview
The project introduces a novel approach for generating domain-adapted clinical summaries using ontology-guided constrained decoding and concept pruning. The method leverages medical ontologies (e.g., SNOMED-CT) to improve the relevance and reduce hallucinations in large language model outputs when summarizing clinical notes.

# Key Features
- Ontology-guided beam search decoding
- Domain-specific clinical text summarization
- Information extraction from Electronic Health Records (EHRs)
- Evaluation framework using the Prometheus model

# Dataset and Ontology
The main dataset used in this project is [MIMIC-III](https://physionet.org/content/mimiciii/1.4/). However, this process could be applied on any dataset of clinical notes. 

# Core Method
The ontology-constrained decoding process works by:

- Computing hierarchy, property, and similarity scores for each beam candidate
- Using diverse beam search to maintain output variety
- Generating domain-adapted summaries through ontology-guided extraction and concept pruning
