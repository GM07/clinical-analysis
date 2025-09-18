# Ontology-Constrained Clinical Text Summarization
This repository contains the implementation of my master's thesis research at Polytechnique Montreal, which resulted in the paper "Ontology-Constrained Generation of Domain-Specific Clinical Summaries" published at EKAW 2024 as well as another preprint paper titled "MedHal: An Evaluation Dataset for Medical Hallucination Detection". The latter resulted in the addition of three new datasets on [Huggingface](https://huggingface.co/datasets/GM07/medhal).

# Ontology-Constrained Generation of Domain Specific Clinical Summaries

## Overview
The project introduces a novel approach for generating domain-adapted clinical summaries using ontology-guided constrained decoding and concept pruning. The method leverages medical ontologies (e.g., SNOMED-CT) to improve the relevance and reduce hallucinations in large language model outputs when summarizing clinical notes. Plus, it develops a new dataset specifically designed for evaluating and improving medical hallucinations detection capabilities of LLMs.

## Key Features
- Ontology-guided beam search decoding
- Domain-specific clinical text summarization
- Information extraction from Electronic Health Records (EHRs)
- Evaluation framework using the Prometheus model

## Dataset and Ontology
The main dataset used in this project for the extraction/summarization task is [MIMIC-III](https://physionet.org/content/mimiciii/1.4/). However, this process could be applied on any dataset of clinical notes.

## Core Method
The ontology-constrained decoding process works by:

- Computing hierarchy, property, and similarity scores for each beam candidate
- Using diverse beam search to maintain output variety
- Generating domain-adapted summaries through ontology-guided extraction and concept pruning

## Citation
If you find this work useful, please cite us in your work as follows:
```
@misc{mehenni2024ontologyconstrainedgenerationdomainspecificclinical,
      title={Ontology-Constrained Generation of Domain-Specific Clinical Summaries}, 
      author={Gaya Mehenni and Amal Zouaq},
      year={2024},
      eprint={2411.15666},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.15666}, 
}
```

# MedHal: An Evaluation Dataset for Medical Hallucination Detection

## Overview
This dataset was created to benchmark LLMs on detecting hallucinated content in clinical settings. It regroups 4 tasks (QA, NLI, Summarization, Information Extraction) all centered around multiple clinical documents (clinical trials, clinical notes, medical questions and scientific papers).

## How are LLMs evaluated ?
LLMs are tasked to evaluate if a statement is factual or not. In order for them to answer YES, everything information mentioned in the statement must be backed up by general medical knowledge or by the provided context.

## Dataset variations
We release three variations of the dataset:
- [MedHal](https://huggingface.co/datasets/GM07/medhal): Raw, unfiltered, unbalanced dataset of over 800k samples
- [MedHal-LF](https://huggingface.co/datasets/GM07/medhal-lf): Length-filtered dataset (the context and statement's lengths summed are always less than 30000 characters which fits inside the 8192 context length window of most models)
- [MedHal-LF-BAL](https://huggingface.co/datasets/GM07/medhal-lf-bal): Length-filtered and task-balanced dataset. This dataset is also splitted into train/val/test.

## Columns
- **id** : Id of the row
- **context** : Context (optional) onto which the statement refers to
- **statement** : Statement (can be factual or not factual)
- **label** : Whether the statement is factual or not (True or False)
- **explanation** : Explanation of why the statement is not factual
- **inner_id** : Id in the original dataset of the row that was used to generate this sample
- **source** : Dataset used to generate this sample :
  - sumpubmed: SumPubMed
  - medmcqa : MedMCQA
  - medqa : MedQA
  - mednli : MedNLI
  - acm : [Augmented-clinical-notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes)
- **synthetic** : Whether the original dataset was a synthetic dataset or not (can be useful to evaluate the impact of synthetic chaining)

## Note
As MedNLI is a semi-private dataset, we removed the samples coming from MedNLI in this version. However, it is pretty easy to create the samples as the task is similar (premise -> context, hypothesis -> statement). Refer to the [paper](https://arxiv.org/pdf/2504.08596) for more information on how MedNLI samples are created. 

## Citation
If you find this dataset useful in your work, please cite the dataset as follows: 
```
@misc{mehenni2025medhalevaluationdatasetmedical,
      title={MedHal: An Evaluation Dataset for Medical Hallucination Detection}, 
      author={Gaya Mehenni and Amal Zouaq},
      year={2025},
      eprint={2504.08596},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.08596}, 
}
```
