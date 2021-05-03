# Overview

This document provides a key checklist for manuscript preparation. Each header below describes suggested content for a single individual paragraph in the overall manuscript. In preparing a manuscript, it is often useful to start with the outlined headers below and populate several bullet points for each paragraph. As needed use the question prompts to help organize your thoughts.

# Abstract

* concise (~250 words) summary of project
* includes all the key headers shown below: introduction, methods, results, conclusion

# Introduction

### Overview of medical background

Why is the disease / problem / task important (2-3 sentences)?

* disease incidence and/or prevalence (in US, world)
* disease morbidity and/or mortality

What could be done to improve current diagnosis (2-3 sentences)?

* increased speed of diagnosis? ==> most common
* increased diagnostic accuracy? ==> less common
* increased reproducibility? 
* increased quantitative / objective data? ==> segmentation, regression networks

**Note**: should try to find several citations where current care *falls short*.

### Overview of deep learning background

What is deep learning (1-2 sentences)?

How has deep learning been used in similar medical applications so far (1-2 sentences)?

What are the limitations of current publications (1-2 sentences)?

* small dataset size?
* simple architecture?
* annotation quality?
* overall performance?

### Summary of project

What is different about the current proposed study (1 sentence)?

What are the key steps / approach of this study (2-3 sentences)?

What are the expected outcomes / hypothesis of this study (1-2 sentences)?

# Methods and Materials

### Data

Where did the dataset come from? Institution-specific or open-source? (1-2 sentences)

How were the patients selected? (2-3 sentences)

* definition of positive and negative cohorts (ground-truth) 
* date ranges (start, stop)
* contiguous or non-contiguous patient selection

How were patients excluded, if any? (1-2 sentences)

How were train / valid / test splits determined? (2-3 sentences)

* important to emphasize that patients with repeat exams were placed into *identical* cross-validation folds, if relevant

### Annotation

How was the data annotated?

* software
* annotation type(s) (global, localization, segmentation)
* number of readers
* consensus metrics (if more than one reader)
* time required for annotation

### Image preprocessing

How was the data preprocessed, if any?

* zero-padding for square matrix (if any)
* resampled to fixed XY-dimension matrix shape (if any)
* resampled to fixed Z-dimension slice thickness (if any)
* normalization scheme
* data augmentation (if any): affine transformations, normalization
* stratified sampling (if any)

### CNN architecture / Experimental design

*Experiment overview (1 paragraph)* 

* describe overall goal of experiments (1 sentence)
* describe each individual model / experiment (2-3 sentences)
* summarize the total number of experiments (1 sentence)
* summarize any common shared features between all models (e.g. common backbone) (1-2 sentences)

*Specific experiments (each subsequent paragraph)*

* use figure(s) for relevant architectures
* describe model output(s) (1 sentence)
* describe model input(s) (1 sentence)
* describe definition of a convolutional block: 3x3 conv > norm > ReLU (1 sentence)
* describe subsampling strategy (e.g. strided conv, pool, etc) (1 sentence)
* describe total number of blocks / layers (1 sentence)
* describe transition to FC (e.g. flatten, average pool, etc) (1 sentence)
* describe fully convolutional layers (if any) (1 sentence)

**Note**: use these paragraphs to describe unique features of each experiment only. For any shared features, move to preceding paragraph(s).

### Implementation

How was the CNN architecture trained?

* describe loss function (1-2 sentences)
* describe loss function modifiers if any (class weights, mask) (1-2 sentences)
* describe optimizer type and learning rate (1 sentence)
* describe batch size and number of training iterations (1 sentence)
* describe weight initialization (1 sentence) ==> Xavier norm (1 sentence)

What type of software and hardware was used?

* Python and Tensorflow / Keras versions (1 sentence)
* GPU(s) models and number (1 sentence)
* estimate time for single experiment (hours) (1 sentence)
* estimate time for single inference prediction (seconds) (1 sentence)

### Statistics

What statistics were analyzed? 

* describe evaluation cohorts (cross-validation, independent test) (1 sentence)
* describe evaluation metric(s) (1 sentence)
  + binary ==> accuracy, sensitivity, specificity, PPV, NPV
  + other ==> Dice, MAE, MSE 
* describe evaluation metric variance (1 sentence)
  + binary ==> AUROC, AUPRC
  + other ==> ==> mean, median, IQR 
* consider evaluation of human-human vs. human-CNN if relevant 

# Results

### Summary of data 

Summarize descriptive statistics about data cohort (1 paragraph)

* total number of patients and exams in each cohort (1-2 sentences)
* distribution of ground-truth (size, distance, etc) (1-2 sentences)

### Summary of results

* summarize results (1 paragraph per experiment; use same headers as `CNN architecture / Experimental design` above)
* create tables / figures to summarize key findings

# Discussion

### Summary of results

* objective summary of key results (1-2 sentences) ==> what are the results?
* interpret key results with relation to hypothesis (1-2 sentences) ==> what do the results imply?
* emphasize any key novel or breakthrough findings (1-2 sentences)

### Previous work

* summarize deep learning work in same organ system / modality (1-2 sentences)
* summarize machine learning work in the same disease entity (2-3 sentences) 
* for each study, emphasize key differences and improvements

### Interesting conclusion(s) (at least 3+ paragraphs)

* state interesting conclusion / observation / trend (1 sentence)
* summarize experimental data supporting claim (2-3 sentences)
* summarize previous work if any supporting claim (1-3 sentences)
* summarize implications of this observation (1-2 sentences)

### Limitations and future work

* address any assumptions made in study design
* sample size / number of participating hospitals or institutions
* quality of ground-truth (if subjective)
* generalizability to similar applications

# Conclusion

What are the key conclusions from the study? (3-4 sentences)

* summary of key results (1-2 sentences)
* summary of key conclusion(s) (1-2 sentences)
* summary of key implications (1-2 sentences)
