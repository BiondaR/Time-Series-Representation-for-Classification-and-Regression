#########################################################################################################
####                                    BACHARELADO EM CIÊNCIAS DA COMPUTAÇÃO                        ####
####                                           TRABALHO DE GRADUAÇÃO                                 ####
####                                            ALUNA: BIONDA ROZIN                                  ####
####                                                25/02/2022                                       ####
#########################################################################################################

# Extrator de features ShapeletTransform
# Referências para uso em
# https://pyts.readthedocs.io/en/stable/modules/transformation.html
# https://pyts.readthedocs.io/en/stable/generated/pyts.transformation.ShapeletTransform.html#pyts.transformation.ShapeletTransform

# Imports
import numpy as np
from pyts.transformation import ShapeletTransform

def ShapeletTransformFeatures(x_train, y_train, x_test, dataset, path):
    shapelet = ShapeletTransform(window_sizes=np.arange(10, 130, 3), random_state=42)
    
    X_train_new = shapelet.fit_transform(x_train, y_train)
    X_test_new = shapelet.transform(x_test)

    features = X_train_new + X_test_new

    np.save(path+dataset+"Shapelet.npy", np.asarray(features))