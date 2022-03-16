#########################################################################################################
####                                    BACHARELADO EM CIÊNCIAS DA COMPUTAÇÃO                        ####
####                                           TRABALHO DE GRADUAÇÃO                                 ####
####                                            ALUNA: BIONDA ROZIN                                  ####
####                                                25/02/2022                                       ####
#########################################################################################################

# Imports
import sys
sys.path.insert(1, 'Features Extractors//')
import feature_extraction as f
import eval as e

# Pipeline de extração de features e posterior classificação

f.extract()
e.runEval()