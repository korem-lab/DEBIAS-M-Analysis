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
data <- read.csv('tmp-dataset-with-added-bias/data.csv', row.names=1)
md <- read.csv('tmp-dataset-with-added-bias/md.csv', row.names=1)

md$Covariate <- md$label

## COMBAT
adjusted <- ComBat_seq( 1 + t(data) %>% as.matrix , batch=md$Study, group=NULL)
### WRITE OUTPUT 
data.frame(adjusted) %>% t() %>% write.csv('tmp-dataset-with-added-bias/combat-out.csv')


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
taxa_corrected1 %>% write.csv('tmp-dataset-with-added-bias/conqur-out.csv')





## MMUPHin
md[, 'mmup_batch'] <- as.factor(md$Study)

mmup_out <- MMUPHin::adjust_batch(t(data/rowSums(data)), 
                                  'mmup_batch', 
                                  data=md
                                  )
t(mmup_out$feature_abd_adj) %>% write.csv('tmp-dataset-with-added-bias/MMUPHin_out.csv')



### PLSDAbatch
data <- data[, colSums(data)>0]
pls_out <- PLSDA_batch(data/rowSums(data),
                       Y.bat=md$Study,
                       near.zero.var=F
                       )

write.csv( pls_out$X.nobatch, 'tmp-dataset-with-added-bias/PLSDAbatch_out.csv')



# ### percentile
#
perc_norm_out <- percentile_norm(data / rowSums(data),
                                 md$Study,
                                 trt=md$Covariate == getmode(md$Covariate),
                                 ctrl.grp=T)

perc_norm_out %>% write.csv('tmp-dataset-with-added-bias/percnorm_out.csv')





### voom-anm

# data <- read.csv('tmp-dataset-with-added-bias/data.csv', row.names=1) %>% as.matrix
# md <- read.csv('tmp-dataset-with-added-bias/md.csv', row.names=1)
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


snmData %>% write.csv('tmp-dataset-with-added-bias/voom-snm-out.csv')

### QUIT ###
q(status=0)






