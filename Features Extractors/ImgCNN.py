#########################################################################################################
####                                    BACHARELADO EM CIÊNCIAS DA COMPUTAÇÃO                        ####
####                                           TRABALHO DE GRADUAÇÃO                                 ####
####                                            ALUNA: BIONDA ROZIN                                  ####
####                                                25/02/2022                                       ####
#########################################################################################################

# Extrator de features baseado em Wang, Z. and Oates, T. (2015a). Encoding time series as images for visual inspection and
# classification using tiled convolutional neural networks. InWorkshops at the twenty-ninth AAAI conference on artificial intelligence.
# Primeiro, é gerada uma representação visual da série temporal; depois, são extraídas características dessas imagens utilizando deep learning

# Imports
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
import matplotlib.pyplot as plt
import shutil
import os

# Talvez usar outras CNNs
def features_img(x_train, x_test, dataset, path, type):
    if type == "GAF":
        # Criação de um objeto GAF
        obj = GramianAngularField()
    elif type == "MTF":
        # Criação de um objeto MTF
        obj = MarkovTransitionField()
    elif type == "RP":
        # Criação de um objeto RP
        obj = RecurrencePlot()

    # Diretório auxiliar para armazenamento das imagens geradas
    dir = "D:\\Downloads\\ESTG\\TCC\\Features Extractors\\images"
    os.mkdir(dir)
    os.chdir(dir)

    # Geração das imagens
    count = 0
    X_1 = obj.fit_transform(x_train)
    X_2 = obj.fit_transform(x_test)

    for i in X_1:
        plt.imshow(i, cmap='rainbow', origin='lower', vmin=-1., vmax=1.)
        plt.savefig(str(count)+".jpg")
        count+=1

    for i in X_2:
        plt.imshow(i, cmap='rainbow', origin='lower', vmin=-1., vmax=1.)
        plt.savefig(str(count)+".jpg")
        count+=1

    os.chdir("D:\\Downloads\\ESTG\\TCC\\Features Extractors\\CNNs\\pretrained-models.pytorch\\pretrainedmodels")

    CNNs = ['resnet152']

    for c in CNNs:
        # Extração das features
        os.system("python extract.py \"" + dir +"\" " + c + " " + dataset + c + type + ".npy")

        shutil.move(dataset + c + type +".npy", path)

    os.chdir("D:\\Downloads\\ESTG\\TCC\\Features Extractors\\images")

    # Deleção das imagens criadas e do diretório auxiliar
    for file in os.listdir():
        os.remove(file)

    os.chdir("D:\\Downloads\\ESTG\\TCC\\Features Extractors")
    os.rmdir("D:\\Downloads\\ESTG\\TCC\\Features Extractors\\images")