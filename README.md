# GENO: negative autoregulation to minimize collateral activity of Cas13d

<img align="right" src="graphical_abstract-01.png" alt="01-spot-overlay" width=300 border="1">

## Description

CRISPR-Cas13d is an programmable RNA-guided RNA endonuclease and a promising candidate for knockdown of RNA in mammalian cells in the lab and the clinic. However, binding of Cas13d to the target RNA unleashes non-specific cleavage of bystander RNAs, or collateral activity, which may confound RNA targeting experiments and raises concerns for Cas13d therapies. Although well appreciated in biochemical and bacterial contexts, the extent of collateral activity in mammalian cells remains disputed.

In this work, we investigated Cas13d collateral activity in the context of an RNA-targeting therapy for DM1, a disease caused by a transcribed long CTG repeat expansion. We found stark evidence of collateral activity when targeting expanded repeats and overexpressed transgenes, and we showed that collateral activity is cytotoxic even when targeting moderately-to-highly expressed endogenous genes. We introduced GENO, a negative autoregulation strategy that selectively minimizes the effects of Cas13d collateral activity when targeting repeat expansions. We recommend thorough screening of collateral activity when applying Cas13d to any target in mammalian cells.

In this study, we used Python for microscopy analysis, alternative splicing quantitation, and dynamical modeling of GENO. The code developed in this study is available here. The manuscript describing this work is available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.12.20.473384v1).

## Navigation
- [Splicing analysis (capillary electrophoresis)](splicing_analysis)
- [Microscopy image analysis (FISH, HCR, IF)](fish_if_analysis)
- [Dynamical modeling of GENO](dynamical_modeling)

[![DOI](https://zenodo.org/badge/439205008.svg)](https://zenodo.org/badge/latestdoi/439205008)
