import argparse
import Benchmarking


def main():
    parser = argparse.ArgumentParser(description="Q-SVM Training")
    parser.add_argument('--U',type=str,default='Shallow_CRZ',help="QSVM Ansatz")
    parser.add_argument('--rM',type=str,default='pca',help="Data Reduction Method")
    parser.add_argument('--q',type=int,default=8,help="Number of Qubits")
    parser.add_argument('--L',type=int,default=1,help="Number of Layers")
    parser.add_argument('--em', type=str, default='Angle_X', help='Embedding Method')
    parser.add_argument('--data', type=str, default='sklearn_make_class', help='Dataset')
    
    args = parser.parse_args()
    print(f"Arguments Received : {args}")
    Configurations = [[args.U, args.rM, args.q, args.L, args.em]]
    data_gen = args.data
    
    
    # Configuration = [[Layer Type, Embedding Type, Qbit number, number of layers,Encoding methods]]
    """Configurations = [#['QAOA', 'pca', 4, 4, 'QAOA'], #Need to Fix Strong Entangle
                  ['Shallow_CRZ', 'no_redu', 8, 1, 'Angle_X'],
                  #['Shallow_CRZ', 'pca', 5, 1, 'Angle_X'],
                 ]
    data_gen = 'capital1_synthetic_data'"""
    Benchmarking.Benchmarking(Configurations,data_gen)  
    
if __name__ == "__main__":
    main()