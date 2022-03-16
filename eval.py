#########################################################################################################
####                                    BACHARELADO EM CIÊNCIAS DA COMPUTAÇÃO                        ####
####                                           TRABALHO DE GRADUAÇÃO                                 ####
####                                            ALUNA: BIONDA ROZIN                                  ####
####                                                25/02/2022                                       ####
#########################################################################################################

# Execução e avaliação de tarefas de classificação

# Imports
from sklearn.linear_model import LinearRegression as LR
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import r2_score as r2
from sklearn.svm import SVC
from sklearn.svm import SVR
import pyts.datasets as dt
import numpy as np

def runEval():
    # Arquivo log para reportar resultados
    log = open("log.txt", "w")

    # Datasets
    datasets_UEA_UCR = ["Gunpoint", "BME", "CBF", "Wine", "Beef", "Plane", "Coffee", "Strawberry"]

    # Features
    features = ["Shapelet", "BeamAngleStatistics", "resnet152GAF", "resnet152MTF", "resnet152RP"]

    path = "D:\\Downloads\\ESTG\\TCC\\Arquivos\\"

    # Métodos de Classificação
    SVMC = SVC(kernel="poly", degree=2, gamma=0.001, C=10)
    MP = MLP()

    # Métodos de Regressão
    SVMR = SVR(kernel="poly", degree=2, gamma=0.001, C=10)
    L = LR()
    
    for d in datasets_UEA_UCR:
        print("Processando o dataset", d)
        print("############# Dataset:", d, "#############", file=log)
        for f in features:
            print("         Feature:", f, file=log)

            feature = np.load(path+d+f+".npy")

            if d == "Coffee":
                _, _, y_train, y_test  = dt.load_coffee(return_X_y=True)
            elif d == "Gunpoint":
                xt, xtt, y_train, y_test  = dt.load_gunpoint(return_X_y=True)
            else:
                _, _, y_train, y_test  = dt.fetch_ucr_dataset(d, return_X_y=True)

            x_train = feature.tolist()[:len(y_train)]
            x_test = feature.tolist()[len(y_train):]

            print("             Classificação", file=log)

            print("                 * SVC", file=log)
            SVMC.fit(x_train, y_train)
            clf = SVMC.predict(x_test)
            print("                     Acurácia:",round(acc(y_test, clf)*100,2),file=log)

            print("                 * MLP",file=log)
            mlp_acc = np.zeros(10)
            for i in range(10):
                MP.fit(x_train, y_train)
                clf2 = MP.predict(x_test)
                mlp_acc[i] = acc(y_test, clf2)
            print("                     Acurácia:",round(np.mean(mlp_acc)*100,2),"+-",round(np.std(mlp_acc)*100,2),file=log)

            print("             Regressão", file=log)

            print("                 * LR",file=log)
            L.fit(x_train, y_train)
            reg = L.predict(x_test)
            print("                     MSE:",round(mse(y_test, reg)*100,2),file=log)
            print("                     R2:",round(r2(y_test, reg)*100,2),file=log)

            print("                 * SVR",file=log)
            SVMR.fit(x_train, y_train)
            reg2 = SVMR.predict(x_test)
            print("                     MSE:",round(mse(y_test, reg2)*100,2),file=log)
            print("                     R2:",round(r2(y_test, reg2)*100,2),file=log)