import QSVM_circuit
import pennylane as qml
from pennylane import numpy as np
import numpy 
import tensorflow as tf
from sklearn.svm import SVC
import autograd.numpy as anp
from sklearn.metrics import hinge_loss

def cross_entropy(labels, predictions):
    loss = 0
    for l,p in zip(labels, predictions):
        c_entropy = l * (anp.log(p)) + (1 - l) * anp.log(1 - p)
        print(c_entropy)
        loss = loss + c_entropy
    return -1 * loss

def hinge_loss_fn(labels, predicitons):
    return -1*hinge_loss(labels,predicitons)
    
def cost_cross_entropy(params, X_init, Y_init, X_t, Y_t,nqubits, layer_type, layer_num, embed):
    #print(params)
    #params = params.reshape(layer_num,9,nqubits)
    qkernel = lambda x1,x2: QSVM_circuit.QSVM_circuit(x1, x2, params, nqubits, layer_type, layer_num, embed)
    qkernel_svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, qkernel))
    qkernel_svm.fit(np.asarray(X_init),Y_init)
    predict = qkernel_svm.predict(X_t)
    print(predict)
    print(Y_t)
    return hinge_loss_fn(Y_t, predict)



def SVC_Loss(kmatrix, labels):
    
    kmatrix = np.array(kmatrix)
    svc = SVC(kernel="precomputed")
    svc.fit(qml.math.toarray(kmatrix), labels)
    dual_coeff = svc.dual_coef_[0]
    support_vec = svc.support_
    kmatrix = kmatrix[support_vec,:][:,support_vec]
    loss = np.sum(np.abs(dual_coeff)) - (0.5 * (dual_coeff.T @ kmatrix @ dual_coeff))
    return -loss
    
itter = 600
learning_rate = 0.01
batch_size = 25
def qsvm_training(X_train,Y_train,nqubits,layer_type,layer_num,embed): 
    X_init = X_train[:int(len(X_train)*0.25)]
    Y_init = Y_train[:int(len(Y_train)*0.25)]
    if layer_type == 'Strong_Entangle':
        total_params = layer_num * nqubits * 3
        params = np.random.uniform(0,2*np.pi, total_params, requires_grad = True)
        #params = tf.reshape(tf.convert_to_tensor(params),(layer_num,nqubits,3))
    elif layer_type == 'Shallow_CRZ':
        params = np.random.uniform(0,2*np.pi,(layer_num,3,nqubits),requires_grad = True)
    elif layer_type == 'Shallow_CRX':
        params = np.random.uniform(0,2*np.pi,(layer_num,3,nqubits),requires_grad = True)
    elif layer_type == 'Deep_Entangle':
        params = np.random.uniform(0,2*np.pi,(layer_num,9,nqubits),requires_grad = True)
    elif layer_type == 'QAOA':
        shape = qml.QAOAEmbedding.shape(n_layers=layer_num, n_wires=nqubits)
        params = np.random.uniform(0,2*np.pi, shape, requires_grad = True)
        
    
        
    #opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    #opt = qml.AdamOptimizer()
    alignment_history = []
    #init_kernel = lambda x1,x2: QSVM_circuit.QSVM_circuit(x1, x2, params, nqubits, layer_type, layer_num, embed)
    #init_kernel_matrix = qml.kernels.square_kernel_matrix(X_init, init_kernel)
    
    #print(f"Initial Square Kernel Matrix: {init_kernel_matrix}")
    #subset = np.random.choice(list(range(len(X_init))), batch_size)
    
    for it in range(itter):
        subset = np.random.choice(list(range(len(X_train))), batch_size)
        
        """  print(type(params))
        params, cost_new = opt.step_and_cost(lambda v: cost_cross_entropy(v, X_train, Y_train, X_train[subset], Y_train[subset],  nqubits, layer_type, layer_num, embed), params.reshape(-1))
        print(cost_new)
        """
        """ params, cost_new = opt.step_and_cost(lambda _params: SVC_Loss(qml.kernels.square_kernel_matrix(
                                                                        X_train[subset], 
                                                                        lambda x1,x2: QSVM_circuit.QSVM_circuit(x1,x2,(_params).reshape(layer_num,3,nqubits),
                                                                                                                nqubits,layer_type,layer_num, embed)
                                                                        ), Y_train[subset]), params.reshape(-1))"""
    
        params,cost = opt.step_and_cost(lambda _params: -qml.kernels.target_alignment(
            X_train[subset],
            Y_train[subset],
            lambda x1, x2:  QSVM_circuit.QSVM_circuit(x1, x2, _params, nqubits, layer_type, layer_num, embed), assume_normalized_kernel=True),params) 
        
        #cost_new = cost_cross_entropy(params, X_init, Y_init, X_train[subset], Y_train[subset],nqubits, layer_type, layer_num, embed)

        if (it + 1) % 10 == 0:
            """  
            current_alignment = qml.kernels.target_alignment(
                X_train, Y_train,lambda x1, x2:  QSVM_circuit.QSVM_circuit(x1, x2, params, nqubits, layer_type, layer_num, embed), assume_normalized_kernel=True
            )
            alignment_history.append(current_alignment)"""
            print(f"Target Alignment Cost:{cost}")
            print(params)
            
            #print(f"Step {it+1} - Alignment = {current_alignment:.3f}")
            
            
            #cost_new = SVC_Loss(qml.kernels.square_kernel_matrix(X_train[subset], lambda x1,x2: QSVM_circuit.QSVM_circuit(x1,x2,params, nqubits,layer_type,layer_num, embed)), Y_train[subset])                             
            #print(f"SVC_Loss:{cost_new}")
    
    trained_kernel = lambda x1, x2: QSVM_circuit.QSVM_circuit(x1, x2, params, nqubits, layer_type, layer_num, embed)
    trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)
    
  
    return trained_kernel_matrix, alignment_history, params, itter