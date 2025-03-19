import Benchmarking
# Configuration = [[Layer Type, Embedding Type, Qbit number, number of layers,Encoding methods]]
Configurations = [#['QAOA', 'pca', 4, 4, 'QAOA'], #Need to Fix Strong Entangle
                  ['Shallow_CRZ', 'pca', 4, 2, 'Angle_X'],
                  ['Deep_Entangle', 'pca', 4, 2, 'Angle_X'],
                  ['QAOA', 'pca', 4, 4, 'QAOA'],
                  ['Shallow_CRZ', 'pca', 8, 2, 'Angle_X'],
                  ['Deep_Entangle', 'pca', 8, 2, 'Angle_X']]
data_gen = 'sklearn_make_class'

Benchmarking.Benchmarking(Configurations,data_gen)