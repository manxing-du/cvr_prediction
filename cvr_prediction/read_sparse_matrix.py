import scipy.sparse as sps
import numpy as np

def convert_to_sparse_matrix(data):
    #country_list=[]
    i = 0
    label = np.zeros([len(data), 1])
    mtx = sps.lil_matrix((len(data), 2584))
    for line in data:
        raw_data = line.split(' ')
        label[i, 0] = raw_data[0]
        features = raw_data[2:]
        indices = map(lambda x: x.split(':')[0], features)
        binary_data = map(lambda x: x.split(':')[1], features)
        mtx[i, indices] = binary_data
        i += 1
        #country_list.append(raw_data[5].split(":")[0])


    #return mtx,label,country_list
    return mtx, label
