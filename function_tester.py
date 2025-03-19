import data
import QSVM_circuit
import Training
import Benchmarking
from sklearn.svm import SVC
import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf

print("All import successful") 

def data_creation_check():
    Set_values = [['Strong_Entangle', 'pca', 14, 2, 'Angle_X'],['Strong_Entangle', 'tsne', 16, 2, 'Angle_X'],['Strong_Entangle', 'svd', 16, 2, 'Angle_X'],['Strong_Entangle', 'autoencode', 8, 2, 'Angle_X']]
    data_gen = 'sklearn_make_class'
    for set in Set_values:
        X_train, Y_train, X_test, Y_test, DataID = data.data_load_and_process(q_num = set[2] ,data_gen = data_gen, data_redu=set[1])
        print(f"DataID: {DataID}, Data_red: {set[1]}")
        print(f"X_train : {X_train[:10]}")
        print(f"Y_train : {Y_train[:10]}")
        print(f"X_test : {X_test[:10]}")
        print(f"Y_test : {Y_test[:10]}")
        print("\n---------------------------------------------------------------")
        
def layer_check(layer_type):
    layer_num = 2
    nqubits = 4
    X_train, Y_train, X_test, Y_test, DataID = data.data_load_and_process(q_num =  nqubits, data_gen = 'sklearn_make_class', data_redu='pca')
    if layer_type == 'Strong_Entangle':
        weights = np.random.uniform(0,2*np.pi, layer_num * nqubits * 3, requires_grad = True)
        dev = qml.device("default.qubit", wires = nqubits)
        wires = dev.wires.tolist()
        @qml.qnode(dev)
        def cir_draw(weights):        
            QSVM_circuit.layer1(weights, wires= wires, layer_num=layer_num, nqubits=nqubits)
            return qml.probs(wires=wires)
        
        weights = tf.reshape(tf.convert_to_tensor(weights),(layer_num,nqubits,3))
    
    elif layer_type == 'Shallow_CRZ':
        weights =  np.random.uniform(0,2*np.pi,(layer_num,3,nqubits),requires_grad = True)
        print(f"Weights: {weights}")
        dev = qml.device("default.qubit", wires = nqubits)
        wires = dev.wires.tolist()
        @qml.qnode(dev)
        def cir_draw(weights):        
            QSVM_circuit.layer2(weights, wires, layer_num, nqubits, X_train[0], embed='Angle_X')
            return qml.probs(wires=wires)
        
    elif layer_type == 'Deep_Entangle':
        weights =  np.random.uniform(0,2*np.pi,(layer_num,9,nqubits),requires_grad = True)
        print(f"Weights: {weights}")
        dev = qml.device("default.qubit", wires = nqubits)
        wires = dev.wires.tolist()
        @qml.qnode(dev)
        def cir_draw(weights):        
            QSVM_circuit.layer3(weights, wires, layer_num, nqubits, X_train[0], embed='Angle_X')
            return qml.probs(wires=wires)
        
    
    print(qml.draw(cir_draw, level = "device")(weights))


def ansatz_call_check(layer_type):
    X_train, Y_train, X_test, Y_test, DataID = data.data_load_and_process(q_num = 8,data_gen = 'sklearn_make_class', data_redu='svd')
    layer_num = 2
    nqubits = 8
    
    if layer_type == 'Strong_Entangle':
        weights = np.random.uniform(0,2*np.pi, layer_num * nqubits * 3, requires_grad = True)
        init_kernel = lambda x1,x2: QSVM_circuit.QSVM_circuit(x1,x2,weights,nqubits,'Strong_Entangle',layer_num,'Angle_X')
    elif layer_type == 'Shallow_CRZ':
        weights = np.random.uniform(0,2*np.pi,(layer_num,3,nqubits),requires_grad = True)
        init_kernel = lambda x1,x2: QSVM_circuit.QSVM_circuit(x1,x2,weights,nqubits,'Shallow_CRZ',layer_num,'Angle_X')
    print(qml.kernels.square_kernel_matrix(X_train[:25],init_kernel))
    
    
def training_check(layer_type):
    X_train, Y_train, X_test, Y_test, DataID = data.data_load_and_process(q_num = 4,data_gen = 'sklearn_make_class', data_redu='pca')
    trained_kernel_matrix, alignment_history, params, itter = Training.qsvm_training(np.array(X_train),np.array(Y_train),4,layer_type,3,'QAOA')
    
    print(f"Trained Kernel Matrix {trained_kernel_matrix}")
    print(f"Params {params}")
    print(f"Itter {itter}")
    

### ------ Function Calls ---------
#data_creation_check()
layer_check("Shallow_CRZ")
#ansatz_call_check("Shallow_CRZ")
#training_check(layer_type = "QAOA")