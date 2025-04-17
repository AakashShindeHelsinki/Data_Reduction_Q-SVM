import Training
import data
from sklearn.svm import SVC
import csv
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from sklearnex import patch_sklearn

patch_sklearn()


def evaluation(predictions, labels):
    accuracy = accuracy_score(predictions, labels)
    precision = precision_score(predictions, labels)
    f1 = f1_score(predictions, labels)
    recall = recall_score(predictions, labels)

    results = [accuracy,precision,f1,recall]
    return results

def Benchmarking(Configurations, data_gen):
    for config in Configurations:
        
        X_train, Y_train, X_test, Y_test, DataID = data.data_load_and_process(q_num = config[2] ,data_gen = data_gen, data_redu=config[1])
        
        trained_kernel_matrix, alignment_history, trained_params, itter = Training.qsvm_training(X_train,Y_train,config[2],config[0],config[3],config[4])
        print(f"Trained Parameters {trained_params}")
        trained_kernel_svm = SVC(kernel=trained_kernel_matrix).fit(X_train,Y_train)
        predictions = trained_kernel_svm.predict(X_test)
        evaluation_results = evaluation(predictions, Y_test)
        
        eva_res = ['accuracy','precision','f1','recall']
        for res,eva in zip(eva_res,evaluation_results):
            print(res+" : "+str(eva))
 
        print("Updating CSV ...")
        field_names = ['DataID','Layer Type','DataPreProcessing','QbitNo','Encoding','Training_Itter','OptimisationAlgo','Lossfn','TrainParams','TrainedKernelMatrix','AlignmentHist','Accuracy','Precision','F1','Recall']

        data_file = [{'DataID':DataID,'Layer Type':config[0], 'DataPreProcessing':config[1],'QbitNo':config[2],'Encoding':config[4],'Training_Itter':itter,
                        'OptimisationAlgo':'Nestrov Momentum', 'TrainParams':trained_params,'TrainedKernelMatrix':trained_kernel_matrix,'AlignmentHist':alignment_history,
                        'Accuracy':evaluation_results[0],'Precision':evaluation_results[1],'F1':evaluation_results[2],'Recall':evaluation_results[3]}]

        with open('Results/TESTFILE_QSVM_Binary_Synthetic.csv', 'a') as csvfile: 
            writer = csv.DictWriter(csvfile, fieldnames = field_names) 
            writer.writeheader() 
            writer.writerows(data_file)
        