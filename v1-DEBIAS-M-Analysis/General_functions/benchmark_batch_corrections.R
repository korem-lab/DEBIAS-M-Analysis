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

snmData %>% write.csv('tmp/voom-snm-out.csv')



## MMUPHin
md[, 'mmup_batch'] <- as.factor(md$Study)

mmup_out <- MMUPHin::adjust_batch(t(data), 
                                  'mmup_batch', 
                                  data=md
                                  )
t(mmup_out$feature_abd_adj) %>% write.csv('tmp/MMUPHin_out.csv')



### PLSDAbatch
pls_out <- PLSDA_batch( (data) / rowSums(data),
                       Y.bat=md$Study, 
                       near.zero.var=F
                       )

write.csv( pls_out$X.nobatch, 'tmp/PLSDAbatch_out.csv')



# ### percentile
# ## pernorm sometimes gives NAs if data isn't in relabund format
## since PLSA is built in the same package, I'm assuming that relabund applies there as well
perc_norm_out <- percentile_norm( ( ( data ) / rowSums(data) ),
                                 md$Study, 
                                 trt=md$Covariate == getmode(md$Covariate), 
                                 ctrl.grp=T)
perc_norm_out[is.na(perc_norm_out)] <- 1e-6

perc_norm_out %>% write.csv('tmp/percnorm_out.csv')





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




## MMUPHin
md[, 'mmup_batch'] <- as.factor(md$Study)

mmup_out <- MMUPHin::adjust_batch(t(data), 
                                  'mmup_batch', 
                                  data=md, 
                                  covariates=c('label')
)
t(mmup_out$feature_abd_adj) %>% write.csv('../data/CRC_with_labels/MMUPHin_out.csv')



### PLSDAbatch
pls_out <- PLSDA_batch( (data) / rowSums(data),
                        Y.bat=md$Study, 
                        near.zero.var=F, 
                        Y.trt=md$label
)

write.csv( pls_out$X.nobatch, '../data/CRC_with_labels/PLSDAbatch_out.csv')




# ### percentile
# ## pernorm sometimes gives NAs if data isn't in relabund format
## since PLSA is built in the same package, I'm assuming that relabund applies there as well
perc_norm_out <- percentile_norm( ( ( data ) / rowSums(data) ),
                                  md$Study, 
                                  trt=md$label, 
                                  ctrl.grp=T)
perc_norm_out[is.na(perc_norm_out)] <- 1e-6

perc_norm_out %>% write.csv('../data/CRC_with_labels/percnorm_out.csv')




set.seet(42)
library(compositions)
### PLSDAbatch of the CLR adata
clr_pls_out <- PLSDA_batch( clr( 1e-6 + ( data / rowSums(data) ) ),
                            Y.bat=md$Study %>% as.factor,
                            near.zero.var=F
)
dim(clr_pls_out$X.nobatch)

write.csv( clr_pls_out$X.nobatch, 
           'tmp/CLR-PLSDAbatch_out.csv'
           )



### QUIT ###
q(status=0)






