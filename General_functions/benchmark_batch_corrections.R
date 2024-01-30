#!/usr/bin/env Rscript

###################################################
## R script that can run ConQuR and Combat-seq,  takes an input:
#  1) a path to a .csv file outlining samples' count-based abundances 
#  2) a path to a .csv file outlining samples' metadata and metadata
#  3) a string outlining teh order the run decontaminations in 
#  4) a path to which teh decontaminated sample file will be written
# and write a csv file of the decontaminatied samples to the specified parth
# It is intended for use with the QIIME2 plugin for SCRuB
#
####################################################

####################################################
#             DESCRIPTION OF ARGUMENTS             #
####################################################
#
# 
library("optparse")

# cat(R.version$version.string, "\n")
errQuit <- function(mesg, status=1) { message("Error: ", mesg); q(status=status) }

option_list = list(
  make_option(c("--samples_counts_path"), action="store", default='NULL', type='character',
              help="File path to the .csv file containing sample abundances"),
  make_option(c("--sample_metadata_path"), action="store", default='NULL', type='character',
              help="File path to the .csv file containing sample metadata"),
  make_option(c("--control_order"), action="store", default='NA', type='character',
              help="the order in which control types should be used for contamination removal. Should be inpuuted as a comma-separated list, i.e. 'control blank library prep,control blank extraction control'"),
  make_option(c("--output_path"), action="store", default='NULL', type='character',
              help="File path to store output csv file. If already exists, will be overwritten")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Assign each of the arguments, in positional order, to an appropriately named R variable
inp.samps <- opt$samples_counts_path
inp.metadata <- opt$sample_metadata_path
cont_order <- opt$control_order
if(cont_order=='NA')cont_order <- NA
out.path <- opt$output_path
### VALIDATE ARGUMENTS ###
# Input directory is expected to contain .fastq.gz file(s)
# that have not yet been filtered and globally trimmed
# to the same length.
if(!file.exists(inp.samps)) {
  errQuit("Input sample file does not exist!")
} else {
  if(!file.exists(inp.metadata)) {
    errQuit("Input metadata file does not exist!")
  }
}
# Output files are to be filenames (not directories) and are to be
# removed and replaced if already present.
for(fn in c(out.path)) {
  if(dir.exists(fn)) {
    errQuit("Output filename ", fn, " is a directory.")
  } else if(file.exists(fn)) {
    invisible(file.remove(fn))
  }
}


## LOAD LIBRARIES ###
suppressMessages({library(tidyverse, quietly=TRUE)})
suppressMessages({library(stringr, quietly=TRUE)})
suppressMessages({library(dplyr, quietly=TRUE)})
suppressMessages({library(rlang, quietly=TRUE)})
suppressMessages({library(ConQuR, quietly=TRUE)})
suppressMessages({library(doParallel, quietly=TRUE)})
suppressMessages({library(sva, quietly=TRUE)})


data <- read.csv('tmp/data.csv', row.names=1)
md <- read.csv('tmp/metadata.csv')

dim(data)

## COMBAT
adjusted <- ComBat_seq( 1 + t(data) %>% as.matrix , batch=md$Study, group=NULL)
### WRITE OUTPUT 
data.frame(adjusted) %>% t() %>% write.csv('tmp/combat-out.csv')


getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
## ConQuR
taxa_corrected1 = ConQuR( tax_tab=data , 
                          batchid=as.factor(as.character(md$Study) ),
                          covariates=data.frame( (Covariate = md$Covariate == getmode(md$Covariate) ) %>% 
                                                   replace_na(F) %>% 
                                                   as.numeric ),
                          # batch_ref="Perez-Santiago.2013",
                          batch_ref = as.character( md$Study[1] ) )


### WRITE OUTPUT 
taxa_corrected1 %>% write.csv('tmp/conqur-out.csv')



### SNM
data <- read.csv('tmp/data.csv', row.names=1)
md <- read.csv('tmp/metadata.csv')
row.names(md) <- row.names(data)

data <- data [ , which( ( data > 0 ) %>% colSums > 2 ) ]

snm_normalised <- snm(raw.dat = data %>% as.matrix() %>% t(), 
                      # bio.var = select(md, Covariate),
                      bio.var = data.frame( ( Covariate = md$Covariate == getmode(md$Covariate) ) %>%
                                    replace_na(F) %>% as.numeric ) %>% as.data.frame,
                      # bio.var = select(md, reads) %>% mutate(reads =replace_na(reads, 10000) ),
                      # bio.var = data.frame( ( Covariate = md$lesion == 'No lesion' ) %>%
                      #               replace_na(F) %>% as.numeric ) %>% as.data.frame,
                      adj.var = select(md, Study) %>% as.data.frame,
                      rm.adj=TRUE,
                      verbose = TRUE,
                      diagnose = TRUE)


snm_normalised_df <- data.frame( t(snm_normalised$norm.dat),
                                 row.names = row.names(data) )


colnames(snm_normalised_df) <- colnames(data)

snm_normalised_df %>% write.csv('tmp/snm-out.csv')




### voom-anm
require(limma)
require(edgeR)
require(dplyr)
require(snm)
require(doMC)
require(tibble)
require(gbm)

data <- read.csv('tmp/data.csv', row.names=1)
md <- read.csv('tmp/metadata.csv')
row.names(md) <- row.names(data)

data <- data [ , which( ( data > 0 ) %>% colSums > 2 ) ]


covDesignNorm <- data.frame( ( Covariate = md$Covariate == getmode(md$Covariate) ) %>%
              replace_na(F) %>% as.numeric ) %>% as.data.frame

vdge <- voom(data %>% t, 
              design = covDesignNorm,
              plot = F, 
              save.plot = F, 
              normalize.method="quantile"
              )

# List biological and normalization variables in model matrices
bio.var <- covDesignNorm

adj.var <- select(md, Study) %>% as.data.frame

colnames(bio.var) <- gsub('([[:punct:]])|\\s+','',colnames(bio.var))
colnames(adj.var) <- gsub('([[:punct:]])|\\s+','',colnames(adj.var))
print(dim(adj.var))
print(dim(bio.var))
print(dim(t(vdge$E)))
print(dim(covDesignNorm))

snmDataObjOnly <- snm(raw.dat = vdge$E, 
                      bio.var = bio.var, 
                      adj.var = adj.var, 
                      rm.adj=TRUE,
                      verbose = TRUE,
                      diagnose = TRUE)
snmData <<- t(snmDataObjOnly$norm.dat)

dim(snmData)



snmData %>% write.csv('../data/CRC/voom-snm-out.csv')









### Speifically for CRC, test when diving everyone the labels
data <- read.csv('tmp/data.csv', row.names=1)
md <- read.csv('tmp/metadata.csv')

dim(data)

## COMBAT
adjusted <- ComBat_seq( 1 + t(data) %>% as.matrix , batch=md$Study, group=md$label)
### WRITE OUTPUT 
data.frame(adjusted) %>% t() %>% write.csv('tmp/combat-out-given-labels.csv')


getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
## ConQuR
taxa_corrected1 = ConQuR( tax_tab=data , 
                          batchid=as.factor(as.character(md$Study) ),
                          covariates=md$label,
                          # batch_ref="Perez-Santiago.2013",
                          batch_ref = as.character( md$Study[1] ) )


### WRITE OUTPUT 
taxa_corrected1 %>% write.csv('tmp/conqur-out-given-labels.csv')



snm_normalised <- snm(raw.dat = data %>% as.matrix() %>% t(), 
                      # bio.var = select(md, Covariate),
                      # bio.var = data.frame( ( Covariate = md$Covariate == getmode(md$Covariate) ) %>%
                      #               replace_na(F) %>% as.numeric ) %>% as.data.frame,
                      # bio.var = select(md, reads) %>% mutate(reads =replace_na(reads, 10000) ),
                      bio.var = md %>% select(label),
                        # data.frame( ( Covariate = md$lesion == 'No lesion' ) %>%
                        #                       replace_na(F) %>% as.numeric ) %>% as.data.frame,
                      adj.var = select(md, Study) %>% as.data.frame,
                      rm.adj=TRUE,
                      verbose = TRUE,
                      diagnose = TRUE)


snm_normalised_df <- data.frame( t(snm_normalised$norm.dat),
                                 row.names = row.names(data) )


colnames(snm_normalised_df) <- colnames(data)

snm_normalised_df %>% write.csv('tmp/snm-out-given-labels.csv')




data <- read.csv('tmp/data.csv', row.names=1)
md <- read.csv('tmp/metadata.csv')
row.names(md) <- row.names(data)

data <- data [ , which( ( data > 0 ) %>% colSums > 2 ) ]


covDesignNorm <- data.frame( ( Covariate = md$label  == getmode(md$label) ) %>%
                               replace_na(F) %>% as.numeric ) %>% as.data.frame

vdge <- voom(data %>% t, 
             design = covDesignNorm,
             plot = F, 
             save.plot = F, 
             normalize.method="quantile"
)

# List biological and normalization variables in model matrices
bio.var <- covDesignNorm

adj.var <- select(md, Study) %>% as.data.frame

colnames(bio.var) <- gsub('([[:punct:]])|\\s+','',colnames(bio.var))
colnames(adj.var) <- gsub('([[:punct:]])|\\s+','',colnames(adj.var))

snmDataObjOnly <- snm(raw.dat = vdge$E, 
                      bio.var = bio.var, 
                      adj.var = adj.var, 
                      rm.adj=TRUE,
                      verbose = TRUE,
                      diagnose = TRUE)
snmData <<- t(snmDataObjOnly$norm.dat)

dim(snmData)

snmData %>% write.csv('../data/CRC_with_labels/voom-snm-out.csv')


### QUIT ###
q(status=0)






