#!/usr/bin/env Rscript

###################################################
## R script that can run ConQuR and Combat-seq,  takes an input:
#  1) a path to a .csv file outlining samples' count-based abundances 
#  2) a path to a .csv file outlining samples' metadata and metadata
#


## LOAD LIBRARIES ###
suppressMessages({library(tidyverse, quietly=TRUE)})
suppressMessages({library(stringr, quietly=TRUE)})
suppressMessages({library(dplyr, quietly=TRUE)})
suppressMessages({library(rlang, quietly=TRUE)})
suppressMessages({library(ConQuR, quietly=TRUE)})
suppressMessages({library(doParallel, quietly=TRUE)})
suppressMessages({library(sva, quietly=TRUE)})
suppressMessages({library(MMUPHin, quietly=TRUE)})
suppressMessages({library(PLSDAbatch, quietly=TRUE)})
### voom-anm
library(limma)
library(edgeR)
library(dplyr)
library(snm)
library(doMC)
library(tibble)
library(gbm)


set.seed(42)
data <- read.csv('tmp-datasets-for-R/data.csv', row.names=1)
md <- read.csv('tmp-datasets-for-R/md.csv', row.names=1)

md$Covariate <- md$label

## COMBAT
adjusted <- ComBat_seq( 1 + t(data) %>% as.matrix , batch=md$Study, group=NULL)
### WRITE OUTPUT
data.frame(adjusted) %>% t() %>% write.csv('tmp-datasets-for-R/combat-out.csv')


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
                          batch_ref = as.character( md$Study[1] ) )


### WRITE OUTPUT
taxa_corrected1 %>% write.csv('tmp-datasets-for-R/conqur-out.csv')





## MMUPHin
md[, 'mmup_batch'] <- as.factor(md$Study)

mmup_out <- MMUPHin::adjust_batch(t(data/rowSums(data)),
                                  'mmup_batch',
                                  data=md
                                  )
t(mmup_out$feature_abd_adj) %>% write.csv('tmp-datasets-for-R/MMUPHin_out.csv')



### PLSDAbatch
pls_out <- PLSDA_batch( ( data / rowSums(data) )[,c(-76)], ## the element 76 causes an error here..
                       Y.bat=md$Study %>% as.factor,
                       near.zero.var=F
                       )

write.csv( pls_out$X.nobatch, 'tmp-datasets-for-R/PLSDAbatch_out.csv')



# ### percentile
#
perc_norm_out <- percentile_norm(data / rowSums(data),
                                 md$Study,
                                 trt=md$Covariate == getmode(md$Covariate),
                                 ctrl.grp=F
                                 )

perc_norm_out %>% write.csv('tmp-datasets-for-R/percnorm_out.csv')





### voom-anm

# data <- read.csv('tmp-datasets-for-R/data.csv', row.names=1) %>% as.matrix
# md <- read.csv('tmp-datasets-for-R/md.csv', row.names=1)
# row.names(md) <- row.names(data)

# data <- data [ , which( ( data > 0 ) %>% colSums > 2 ) ]


covDesignNorm <- data.frame( ( Covariate = md$Covariate == getmode(md$Covariate) ) %>%
                               replace_na(F) %>% as.numeric ) %>% as.data.frame

vdge <- voom(data %>% t(), 
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

snmData %>% write.csv('tmp-datasets-for-R/voom-snm-out.csv')


set.seet(42)
library(compositions)
### PLSDAbatch of the CLR adata
clr_pls_out <- PLSDA_batch( clr( 1e-6 + ( data / rowSums(data) ) ),
                        Y.bat=md$Study %>% as.factor,
                        near.zero.var=F
)

write.csv( clr_pls_out$X.nobatch, 'tmp-datasets-for-R/CLR-PLSDAbatch_out.csv')


### QUIT ###
q(status=0)






