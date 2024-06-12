import sympy
import numpy as np
import cloudpickle
import pickle
import os.path
import os
import errno
import csv


def new_sym(name):
    return sympy.symbols(name, real=True)


def vec2so3(vec):
    return sympy.Matrix([[0,        -vec[2],    vec[1]],
                         [vec[2],   0,          -vec[0]],
                         [-vec[1],  vec[0],     0]])


def so32vec(mat):
    return sympy.Matrix([[mat[2, 1]],
                         [mat[0, 2]],
                         [mat[1, 0]]])


def inertia_vec2tensor(vec):
    return sympy.Matrix([[vec[0], vec[1], vec[2]],
                         [vec[1], vec[3], vec[4]],
                         [vec[2], vec[4], vec[5]]])


def inertia_tensor2vec(I):
    return [I[0, 0], I[0, 1], I[0, 2], I[1, 1], I[1, 2], I[2, 2]]


def tranlation_transfmat(v):
    return sympy.Matrix([[1, 0, 0, v[0]],
                        [0, 1, 0, v[1]],
                        [0, 0, 1, v[2]],
                        [0, 0, 0, 1]])


def ml2r(m, l):
    return sympy.Matrix(l) / m


def Lmr2I(L, m, r):
    return sympy.Matrix(L - m * vec2so3(r).transpose() * vec2so3(r))


def gen_DLki_mat():
    M = list(range(10))
    for i in range(10):
        M[i] = np.zeros((6, 6))
    # Lxx
    M[0][0, 0] = 1
    # Lxy
    M[1][0, 1] = 1
    M[1][1, 0] = 1
    # Lxz
    M[2][0, 2] = 1
    M[2][2, 0] = 1
    # Lyy
    M[3][1, 1] = 1
    # Lyz
    M[4][1, 2] = 1
    M[4][2, 1] = 1
    # Lzz
    M[5][2, 2] = 1
    # lx
    M[6][1, 5] = 1
    M[6][5, 1] = 1
    M[6][2, 4] = -1
    M[6][4, 2] = -1
    # ly
    M[7][0, 5] = -1
    M[7][5, 0] = -1
    M[7][2, 3] = 1
    M[7][3, 2] = 1
    # lz
    M[8][0, 4] = 1
    M[8][4, 0] = 1
    M[8][1, 3] = -1
    M[8][3, 1] = -1
    # m
    M[9][3, 3] = 1
    M[9][4, 4] = 1
    M[9][5, 5] = 1

    return M


def gen_DLki_mat4():
    M = list(range(10))
    for i in range(10):
        M[i] = np.zeros((4, 4))
    # Lxx
    M[0][0, 0] = -0.5
    M[0][1, 1] = 0.5
    M[0][2, 2] = 0.5

    # Lxy
    M[1][0, 1] = -1
    M[1][1, 0] = -1
    # Lxz
    M[2][0, 2] = -1
    M[2][2, 0] = -1
    # Lyy
    M[3][0, 0] = 0.5
    M[3][1, 1] = -0.5
    M[3][2, 2] = 0.5

    # Lyz
    M[4][1, 2] = -1
    M[4][2, 1] = -1
    # Lzz
    M[5][0, 0] = 0.5
    M[5][1, 1] = 0.5
    M[5][2, 2] = -0.5
    # lx
    M[6][0, 3] = 1
    M[6][3, 0] = 1
    # ly
    M[7][1, 3] = 1
    M[7][3, 1] = 1
    # lz
    M[8][2, 3] = 1
    M[8][3, 2] = 1
    # m
    M[9][3, 3] = 1

    return M


def save_data(folder, name, data):
    model_file = os.path.join(folder, f'{name}.pkl')

    if not os.path.exists(os.path.dirname(model_file)):
        try:
            os.makedirs(os.path.dirname(model_file))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(model_file, 'wb') as f:
        cloudpickle.dump(data, f)

def save_csv_data(folder, name, data):
    file_path = os.path.join(folder, f'{name}.csv')
    with open(file_path, 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        for i in range(np.size(data, 0) - 10):
            wr.writerow(data[i])

def load_data(folder, name):
    model_file = os.path.join(folder, f'{name}.pkl')
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        raise Exception("No {} can be found!".format(model_file))