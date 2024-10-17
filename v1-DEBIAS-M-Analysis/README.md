

# Benchmarking and analysis for DEBIAS-M

Welcome to DEBIAS-M! DEBIAS-M is a computational methods designed to identify and correct for the predominant issue of sequencing bias in microbiome data. In this copy of our Codeocean capsule, we provide full code and data to replicate the analyses from the DEBIAS-M manuscript. 

## Code Walkthrough
The entire analysis from this capsule can be executed by running the `code/run` script, which calls the necessary functions from within the `code` directory in the proper order. We structure the rest of the code directory analysis as follows:
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
| `02-metabolite-plots.py` | Write the metabolite plots; assumes that the `main.py` file has already been executed` |
| `03-adaptation.py` | Code for all our 'adaptation' benchmarks and analyses, as described in the manuscript. |
| `04-bcf-analyses.py` | Code to analyze the fitted bias correction factors ('BCFs') learned when training DEBIAS-M on the HIVRC dataset. |
| `05-Cervix-RF.py` | Code to analyze the cervix microbiome datasets using randm forest models, and write the auROC, auPR, and boxplots |
| `06-make-simulation-plots.py` | Reads the precomputed simluation resulting runs and writes the plots into `results` |
| `07-Run-immunotherapy.py` | Run Linear and random forest benchmarks for prediction of PFS12 response in melanoma patients, using data from Lee et al. |
| `adonis-analysis.Rmd` | Code to run adonis and the R heatmap for inference of fitted BCFs |


## Data
All datasets analyzed in this study are publicly available. The HIV dataset is available from Synapse (https://www.synapse.org/#!Synapse:syn18406854). The colorectal cancer dataset is available through the R curatedMetagenomicData package40. The cervical neoplasia dataset was compiled from data provided with each publication, with information detailed in Table S2. 
All processed data used in our analysess is stored in the `data` directory. 

## Environment
The environment used in this analysis involes python 3.6. We provide a docker file which can be used to set up the environment for a successful capsule run.

## Support
If you have any questions regarding the analyis outlined in this capsule, or run into any issues when using DEBIAS-M, please reach out via our [issues page](https://github.com/korem-lab/DEBIAS-M/issues).

