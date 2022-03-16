#########################################################################################################
####                                    BACHARELADO EM CIÊNCIAS DA COMPUTAÇÃO                        ####
####                                           TRABALHO DE GRADUAÇÃO                                 ####
####                                            ALUNA: BIONDA ROZIN                                  ####
####                                                25/02/2022                                       ####
#########################################################################################################

# Extrator de features Beam Angle Statistics

# Imports
import math
import numpy as np

# Algoritmo para extração de features baseado em formas, Beam Angle Statistics (BAS)
# Dado um valor de k, as features extraídas para cada ponto x é o ângulo entre as retas formadas pelos pontos (x-k, y1), (x, y) e (x, y), (x+k, y2)
# O parâmetro "deg" indica se os ângulos calculados serão representados em graus ou radianos
def beam_angle_statistics(xts, k=30, deg="degrees"):
    t_points = []
    x = 0
    for i in range(len(xts)):
        t_points.append([x, xts[i]])
        x += 1

    feat_vec = []

    for i in range(k, len(xts)-k):
        m1 = (t_points[i][1]-t_points[i-k][1])/(t_points[i-k][0]-t_points[i][0])
        m2 = (t_points[i][1]-t_points[i+k][1])/(t_points[i+k][0]-t_points[i][0])

        if deg == "degrees":
            alpha = math.degrees(math.atan(abs((m2 - m1)/(1+m2*m1))))
        elif deg == "rad":
            alpha = math.atan(abs((m2 - m1)/(1+m2*m1)))

        feat_vec.append(alpha)

    return feat_vec

# Extração das features dos conjunto de treinamento e teste
def features_bas(x_train, x_test, dataset, path, k=30, deg="degrees"):
    features = []

    # Cálculo automático de k
    if k == 'auto':
        list_aux = []
        for i in x_train:
            list_aux.append(np.std(np.asarray(i)))
        for i in x_test:
            list_aux.append(np.std(np.asarray(i)))

        k = np.mean(np.asarray(list_aux))
        const = 0.09
        num = len(x_train[0])
        k = int(round(k*const*num, 1))

    for i in x_train:
        features.append(beam_angle_statistics(i, k, deg))

    for i in x_test:
        features.append(beam_angle_statistics(i, k, deg))

    np.save(path+dataset+"BeamAngleStatistics.npy", np.asarray(features))