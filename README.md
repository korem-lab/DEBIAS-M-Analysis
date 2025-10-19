



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
| make-PR-csv.ipynb | Saves the table describing all auPRs in the main benchmarks (Supplementary Table 2) |
| v1-DEBIAS-M-Analysis | Includes code for all remaining analyses (see nested README within this folder). More details on this directory's structure below |


## Code Walkthrough within `v1-DEBIAS-M-Analysis` directory
The entire analysis from the `v1-DEBIAS-M-Analysis` directory can be executed by running the `code/run` script, which calls the necessary functions from within the `code` directory in the proper order. We structure the rest of the code directory analysis as follows:
| File/Folder | Description |
|--|--|
| main.py | Script that loops through the main tasks to execute the generalizeable functions on all datasets in one command | 
| `SBCMP` | Python scripts involved in the backend of the DEBIAS-M processing |
| `DEBIAS-M` | The DEBIAS-M package, which provides both access to the backend format of the `torch` training functions, and to scikit-learn formatted classes. The package is also available [here](https://github.com/korem-lab/DEBIAS-M). |
| `General_functions` | Includes the code that was applied to each separate dataset, including 1) R script that runs the batch correction methods in R; 2) Generalizeable prediction benchmark code and delong test code; 3) scripts that load each dataset into a similar format; and 4) code to produce plots |
| `Cervix` | Includes the baseline code for analyses involving cervix microbiome data. The two files have the code for 1) the general linear model benchmarksl and 2) the random forest benchmarks|
| `HIVRC` | Includes the code for inference that is run on the HIVRC data. Scripts unique to the HIVRC data focus on the inference of fitted bias correction factors ('BCFs')|
| `CRC` | Code with methods specific for our CRC analyses. Nothing is run from this directory, it only exists becuase the `main.py` file looks for a command specific to each task  |
 `Metabolites` | Code for the metabolite analyses. These files loop through all metabolites to produce predictions, and runs the multitask version of the DEBIAS-M model. Also include the R code and results from training MelonnPan on this dataset |
| `Simulations` |  Code to run the simulations. For brevity, it is not run within this capsule, although the results from running these files are included in `data`. The simulation plots are created in the `06-make-simulation-plots.py` file |
| `01-cervic-roc-plot.py`| script that runs the cervical carcinoma benchmark and writes the ROC plots into the results folder. Also write the raw predictions for a future comparison to the random forest results.   | 
| `02-metabolite-plots.py` | Write the metabolite Fig S8b plot; assumes that the `main.py` file has already been executed` |
| `03-adaptation.py` | Code for all our 'adaptation' benchmarks and analyses, as described in the manuscript. |
| `04-bcf-analyses.py` | Code to analyze the fitted bias correction factors ('BCFs') learned when training DEBIAS-M on the HIVRC dataset. |
| `05-Cervix-RF.py` | Code to analyze the cervix microbiome datasets using randm forest models, and write the auROC, auPR, and boxplots |
| `06-make-simulation-plots.py` | Reads the precomputed simluation resulting runs and writes the plots into `results` |
| `07-Run-immunotherapy.py` | Run Linear and random forest benchmarks for prediction of PFS12 response in melanoma patients, using data from Lee et al. |
| `adonis-analysis.Rmd` | Code to run adonis and the R heatmap for inference of fitted BCFs |

