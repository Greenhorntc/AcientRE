# AcientRE
Hierarchical Relation Extraction for Ancient Classical Texts via Integration of Pre-trained and Large Language Models

We propose a hierarchical relation extraction framework
for classical texts that synergizes pre-trained models with LLMs. This architecture employs a
cascaded pipeline: 1. Relation Candidate Generation, where an enhanced pre-trained model serves
as an initial classifier to produce candidate relation sets for subsequent LLM processing; 2. Fine-
grained Relation Classification, which incorporates modern Chinese paraphrases of classical texts
and formal definitions of candidate relations as contextual knowledge. This stage leverages the
task-solving capabilities of LLMs to infer optimal relations from candidate sets through Chain-of-
Thought (CoT) reasoning. Experimental validation on the CHisIEC dataset demonstrates state-of-the-art performance, achieving 89.92% accuracy surpassing baseline models by +0.44%. Notably, the framework exhibits exceptional few-shot learning capabilities: with merely 50 samples per class, it matches the performance (83.91% accuracy, 77.35% F1) of the fully fine-tuned ChatGLM2-6B model trained on the complete dataset.


# Two-Stage Model Inference Pipeline

This project implements an efficient two-stage inference pipeline to balance performance and computational cost. The workflow consists of the following steps:

## Pipeline Overview

1. **First Stage - Small Model Fine-tuning & Candidate Selection**
   - Fine-tune a smaller lightweight model for initial inference
   - Generate and select the top 4 most promising candidate outputs

2. **Second Stage - Large Model Refinement**
   - Use the selected candidates from stage 1 to initialize the larger model
   - Perform second-stage inference with the large model for final output refinement

## Workflow Diagram
