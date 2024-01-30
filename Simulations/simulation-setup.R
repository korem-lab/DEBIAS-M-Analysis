
library(dplyr)
library(SparseDOSSA2)

load('preSCRuB_SPARSEDOSSA_generators.Rdata')

for( seed in 1:25 ){
  set.seed(seed)
  n_feats <- 1000
  n_samps <- 4*96
  samples <- SparseDOSSA2(template = fitted_sparsedossa_vag_samples, 
                          n_sample=n_samps, 
                          n_feature = n_feats , 
                          median_read_depth = 10000, 
                          verbose = F)$simulated_data %>% t()
  
  samples %>% write.csv(paste0('generated_datasets/dataset-seed-', seed, '.csv'))
  
} 