from pennylane import AngleEmbedding, AmplitudeEmbedding, IQPEmbedding, SqueezingEmbedding, DisplacementEmbedding

def data_embedding(X, q_num, type = 'Angle_X'):
    if type == 'Angle_X':
        AngleEmbedding(X, wires=range(q_num), rotation='X')
    elif type == 'Angle_Y':
        AngleEmbedding(X, wires=range(q_num), rotation='Y')
    elif type == 'Angle_Z':
        AngleEmbedding(X, wires=range(q_num), rotation='Z')
    elif type == 'Amplitude':
        AmplitudeEmbedding(X, wires=range(q_num), normalize=True, pad_with=1)
    elif type == 'IQP':
        IQPEmbedding(X,wires=range(q_num),n_repeats=2)
    