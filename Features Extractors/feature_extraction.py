#########################################################################################################
####                                    BACHARELADO EM CIÊNCIAS DA COMPUTAÇÃO                        ####
####                                           TRABALHO DE GRADUAÇÃO                                 ####
####                                            ALUNA: BIONDA ROZIN                                  ####
####                                                25/02/2022                                       ####
#########################################################################################################
# Imports
import pyts.datasets as dt
import BeamAngleStatistics as bas
import ImgCNN as imc
import ShapeletTransform as st

def extract():
    # Seleção de três datasets das bases de dados UEA e UCR
    datasets_UEA_UCR = ["BME", "Beef", "Plane", "Coffee", "Gunpoint", "CBF", "Strawberry", "Wine"]

    imagens = ["GAF", "MTF", "RP"]

    path = "D:\\Downloads\\ESTG\\TCC\\Arquivos\\"

    # Extração de features
    for d in datasets_UEA_UCR:
        print("Processando o dataset", d, "...")
        if d == "Coffee":
            x_train, x_test, y_train, _  = dt.load_coffee(return_X_y=True)
        elif d == "Gunpoint":
            x_train, x_test, y_train, _  = dt.load_gunpoint(return_X_y=True)
        else:
            x_train, x_test, y_train, _  = dt.fetch_ucr_dataset(d, return_X_y=True)

        for i in imagens:
            print(i,"...")
            imc.features_img(x_train, x_test, d, path, i)
        bas.features_bas(x_train, x_test, d, path, k="auto", deg="degrees")
        st.ShapeletTransformFeatures(x_train, y_train, x_test, d, path)