import pennylane as qml
import Embedding
import tensorflow as tf
import numpy as np

#Layer 
def layer1(params,wires,layer_num,nqubits): #Strongly Entangling layer
    #params = tf.reshape(tf.convert_to_tensor(params),(layer_num,nqubits,3))
    qml.StronglyEntanglingLayers(params,wires) 
    

def layer2(params, wires, layer_num, nqubits, x, embed): #Shallow Entangle CRZ
    for l in range(0,layer_num):
        for  i in range(0,nqubits):
            qml.RZ(params[l,0,i],wires=wires[i])
            qml.RY(params[l,1,i],wires=wires[i])
            
        for i in range(0,nqubits-1):
            qml.CRZ(params[l,2,i],wires=[i,i+1])
        qml.CRZ(params[l,2,nqubits-1],wires=[nqubits-1,0])
        
        for  i in range(0,nqubits):
            qml.Hadamard(wires=wires[i])
        Embedding.data_embedding(x, nqubits,type=embed)
        
def layer3(params, wires, layer_num, nqubits, x, embed): ##Deep Entangling layer
    for l in range(0,layer_num):
        for  i in range(0,nqubits):
            qml.RZ(params[l,0,i],wires=wires[i])
            qml.RY(params[l,1,i],wires=wires[i])
            
        for i in range(0,nqubits-1):
            qml.CRZ(params[l,2,i],wires=[i,i+1])
        qml.CRZ(params[l,2,nqubits-1],wires=[nqubits-1,0])
        
        for  i in range(0,nqubits):
            qml.RX(params[l,3,i],wires=wires[i])
            qml.RY(params[l,4,i],wires=wires[i])
            
        for i in range(0,nqubits-1):
            qml.CRY(params[l,5,i],wires=[i,i+1])
        qml.CRY(params[l,5,nqubits-1],wires=[nqubits-1,0])
        
        for  i in range(0,nqubits):
            qml.Hadamard(wires=wires[i])
            qml.RX(params[l,6,i],wires=wires[i])
            qml.RZ(params[l,7,i],wires=wires[i])
            
        for i in range(0,nqubits-1):
            qml.CRX(params[l,8,i],wires=[i+1,i])
        qml.CRX(params[l,8,nqubits-1],wires=[0,nqubits-1])
        
        for  i in range(0,nqubits):
            qml.Hadamard(wires=wires[i])
        Embedding.data_embedding(x, nqubits,type=embed)
        
def layer4(x, params, nqubits, wires): #QAOA Encoding
    qml.QAOAEmbedding(features=x,weights=params,wires=range(nqubits))
    
    

#Full Ansatz Ciruit
def ansatz(x, params, nqubits, wires, layer_type, layer_num, embed):
    if embed == 'QAOA':
        layer_type == 'QAOA'
        layer4(x, params, nqubits, wires)
    else:
        for  i in range(0,nqubits):
            qml.Hadamard(wires=wires[i])
        Embedding.data_embedding(x, nqubits,type=embed)
        if layer_type == 'Strong_Entangle':
            params = tf.reshape(tf.convert_to_tensor(params),(layer_num,nqubits,3))
            layer1(params,wires,layer_num,nqubits)
        elif layer_type == 'Shallow_CRZ':
            layer2(params,wires,layer_num,nqubits,x,embed)
        elif layer_type == 'Deep_Entangle':
            layer3(params,wires,layer_num,nqubits,x,embed)

#QSVM Circuit call  

def QSVM_circuit(x1, x2, params, nqubits, layer_type, layer_num, embed):
    dev = qml.device("default.qubit", wires = nqubits)
    wires = dev.wires.tolist()

    @qml.qnode(dev)
    def kernel_circuit(x1, x2, params, nqubits, layer_type=layer_type, layer_num=layer_num, embed=embed, wires= wires):
        ansatz(x1, params, nqubits, wires, layer_type, layer_num, embed)
        qml.adjoint(ansatz)(x2, params, nqubits, wires, layer_type, layer_num, embed)
        return qml.probs(wires=wires)
    
    return kernel_circuit(x1,x2,params,nqubits, layer_type=layer_type, layer_num=layer_num, embed=embed, wires= wires)[0]