



# Benchmarking and analysis for DEBIAS-M

Welcome to DEBIAS-M! DEBIAS-M is a computational methods designed to identify and correct for processing bias and batch effects in microbiome data. 

## Code Walkthrough
Every folder within this repository contains analyses to produce a different component within the `results` or `plots` directories. The different analyses contained within each folder are as follows:

| File/Folder | Description |
|--|--|
|CRC-simulations	 |   Code to run and plot the  simulutations synthetically varying different underlying parameters in the colorectal cancer prediction benchmark (Fig S5)|	
| pseudocount-evaluation | Code evaluating the impact of pseudocounts and flooring as pre- and post-processing steps for DEBIAS-M (Fig S10a-c)|
|HIV-using-age	| Code evaluating the use of different covariates during batch-correction on the HIV benchmark (Figs 2a, S3c)|
|regression| The regression benchmark for metabolite prediction (Figs 5c, S8)|
|regression-all-methods| The regression benchmark for the plot incorporating a wider array of batch-correction methods (Fig 5b)|
|Simulations | Implementing the simulations using synthetically generated data (Figs 3, S4, S6) |
|runtime-benchmark| Evaluates DEBIAS-M's runtime under various scenarios (Fig S10h,i)|
|convex-assumption | Generates visualizations describing the optimization space of DEBIAS-M, illustrating non-convex spaces |
| single-batch-cv| Testing DEBIAS-M as a single-study processor correcting for biases while only considering a single batch (Fig S9b)| 
|study-weighting | Evaluates the impact of weighting the influence of each study's impact on the cross-batch loss based on each study's size (Fig R3-4)|
|loss-functions	| Evaluates DEBIAS-M on some of the main benchmarks using a wide variety of loss functions (Fig S10f,g)|	
|taxonomy-aggregation | Evaluates the impact of aggregating microbial features to different levels of taxonomy (Fig S10d,e)|
|new-cervix-carcinoma | Implementation of DEBIAS-M cervical carcinoma analyses, using a newly created combination of cervical datasets (Figs 2d, 6a,b)|
| within-control-comparison | Evaluates a version of Online DEBIAS-M in which similarity is only enforced within the controls of the training samples (Fig R2-2)|
| v1-DEBIAS-M-Analysis | Includes code for all remaining analyses (see nested README within this folder) |
| make-PR-csv.ipynb | Saves the table describing all auPRs in the main benchmarks (Supplementary Table 2) |

