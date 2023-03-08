# from src.cython.slow_loops import problem_18_27, problem_17_26, np_max, sum_cy

# from src.heco_de1.utils import sgn
from numpy import sin, sqrt, cos, pi, exp, e
# import random
import numpy as np
import scipy.io as sio
import os
script_dir = os.path.dirname(__file__)


def load_mat(problem_id, dim):
    o_shift = np.zeros((1, dim), dtype=np.float64)
    matrix = np.zeros((dim, dim), dtype=np.float64)
    matrix1 = np.zeros((dim, dim), dtype=np.float64)
    matrix2 = np.zeros((dim, dim), dtype=np.float64)
    matrix_d = {10: 'M_10', 30: 'M_30', 50: 'M_50', 100: 'M_100'}
    matrix1_d = {10: 'M1_10', 30: 'M1_30', 50: 'M1_50', 100: 'M1_100'}
    matrix2_d = {10: 'M2_10', 30: 'M2_30', 50: 'M2_50', 100: 'M2_100'}
    if problem_id in [1, 3, 4, 6, 7, 8, 9]:
        mat_contents = sio.loadmat(script_dir + '/input_data/Function{}.mat'.format(problem_id))
        o_shift[0, :] = mat_contents['o'][0, :dim]
    elif problem_id == 2:
        mat_contents = sio.loadmat(script_dir + '/input_data/Function2.mat')
        o_shift[0, :] = mat_contents['o'][0, :dim]
        matrix[:] = mat_contents[matrix_d[dim]]
    elif problem_id == 5:
        mat_contents = sio.loadmat(script_dir + '/input_data/Function5.mat')
        o_shift[0, :] = mat_contents['o'][0, :dim]
        matrix1[:] = mat_contents[matrix1_d[dim]]
        matrix2[:] = mat_contents[matrix2_d[dim]]
    elif problem_id in range(10, 21):
        mat_contents = sio.loadmat(script_dir + '/input_data/ShiftAndRotation.mat')
        o_shift[0, :] = mat_contents['o'][0, :dim]
    elif problem_id in range(21, 29):
        mat_contents = sio.loadmat(script_dir + '/input_data/ShiftAndRotation.mat')
        matrix[:] = mat_contents[matrix_d[dim]]
    return o_shift, matrix, matrix1, matrix2


class Cec2017:
    def __init__(self, problem_id, dim, o_shift, matrix, matrix1, matrix2):
        self.problem_id = problem_id
        self.dim = dim
        self.o_shift = o_shift
        self.matrix = matrix
        self.matrix1 = matrix1
        self.matrix2 = matrix2

    def shift_func(self, x):
        y = (x[:self.dim, :].T - self.o_shift[0]).T
        return y

    @staticmethod
    def rotate_func(y, matrix):
        z = matrix.dot(y)
        return z

    def benchmark(self, x):
        y = self.shift_func(x)
        if self.problem_id == 1:
            f = ((np.cumsum(y, axis=0))**2).sum(axis=0)
            v_g1 = (y ** 2 - 5000.0 * cos(0.1 * pi * y) - 4000.0).sum(axis=0)
            v_g1[v_g1 < 0.0] = 0.0
            v = v_g1

        elif self.problem_id == 2:
            z = self.rotate_func(y, self.matrix)
            f = ((np.cumsum(y, axis=0)) ** 2).sum(axis=0)
            v_g1 = (z ** 2 - 5000.0 * cos(0.1 * pi * z) - 4000.0).sum(axis=0)
            v_g1[v_g1 < 0.0] = 0.0
            v = v_g1

        elif self.problem_id == 3:
            f = ((np.cumsum(y, axis=0)) ** 2).sum(axis=0)
            v_g1 = (y ** 2 - 5000.0 * cos(0.1 * pi * y) - 4000.0).sum(axis=0)
            v_h1 = abs((y * sin(0.1 * pi * y)).sum(axis=0)) - 1E-4
            v_g1[v_g1 < 0.0] = 0.0
            v_h1[v_h1 < 0.0] = 0.0
            v = (v_g1 + v_h1) / 2

        elif self.problem_id == 4:
            f = (y**2 - 10.0 * cos(2.0 * pi * y) + 10.0).sum(axis=0)
            v_g1 = -(y * sin(2.0 * y)).sum(axis=0)
            v_g2 = (y * sin(y)).sum(axis=0)
            v_g1[v_g1 < 0.0] = 0.0
            v_g2[v_g2 < 0.0] = 0.0
            v = (v_g1 + v_g2) / 2

        elif self.problem_id == 5:
            w1 = self.rotate_func(y, self.matrix1)
            w2 = self.rotate_func(y, self.matrix2)
            f = (100.0 * (y[:-1]**2 - y[1:])**2
                 + (y[:self.dim - 1] - 1.0)**2).sum(axis=0)
            v_g1 = (w1**2 - 50.0 * cos(2.0 * pi * w1) - 40.0).sum(axis=0)
            v_g2 = (w2**2 - 50.0 * cos(2.0 * pi * w2) - 40.0).sum(axis=0)
            v_g1[v_g1 < 0.0] = 0.0
            v_g2[v_g2 < 0.0] = 0.0
            v = (v_g1 + v_g2) / 2

        if self.problem_id == 6:
            f = (y**2 - 10.0 * cos(2.0 * pi * y) + 10.0).sum(axis=0)
            v_h1 = abs((-y * sin(2.0 * y)).sum(axis=0)) - 1E-4
            v_h2 = abs((y * sin(2.0 * pi * y)).sum(axis=0)) - 1E-4
            v_h3 = abs((-y * cos(2.0 * y)).sum(axis=0)) - 1E-4
            v_h4 = abs((y * cos(2.0 * pi * y)).sum(axis=0)) - 1E-4
            v_h5 = abs((y * sin(2.0 * (abs(y)) ** 0.5)).sum(axis=0)) - 1E-4
            v_h6 = abs((-y * sin(2.0 * (abs(y)) ** 0.5)).sum(axis=0)) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_h2[v_h2 < 0.0] = 0.0
            v_h3[v_h3 < 0.0] = 0.0
            v_h4[v_h4 < 0.0] = 0.0
            v_h5[v_h5 < 0.0] = 0.0
            v_h6[v_h6 < 0.0] = 0.0
            v = (v_h1 + v_h2 + v_h3 + v_h4 + v_h5 + v_h6) / 6

        elif self.problem_id == 7:
            f = (y * sin(y)).sum(axis=0)
            temp = y - 100.0 * cos(0.5 * y) + 100.0
            v_h1 = abs(temp.sum(axis=0)) - 1E-4
            v_h2 = abs((-temp).sum(axis=0)) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_h2[v_h2 < 0.0] = 0.0
            v = (v_h1 + v_h2) / 2

        elif self.problem_id == 8:
            f = np.amax(y, axis=0)
            y_odd = y[::2, :]
            y_even = y[1::2, :]
            v_h1 = abs((np.cumsum(y_odd, axis=0) ** 2).sum(axis=0)) - 1E-4
            v_h2 = abs((np.cumsum(y_even, axis=0) ** 2).sum(axis=0)) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_h2[v_h2 < 0.0] = 0.0
            v = (v_h1 + v_h2) / 2

        elif self.problem_id == 9:
            f = np.amax(y, axis=0)
            y_odd = y[::2, :]
            y_even = y[1::2, :]
            v_g1 = np.prod(y_even, axis=0)
            v_h1 = abs(((y_odd[:-1] ** 2 - y_odd[1:]) ** 2).sum(axis=0)) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_g1[v_g1 < 0.0] = 0.0
            v = (v_h1 + v_g1) / 2

        elif self.problem_id == 10:
            f = np.amax(y, axis=0)
            v_h1 = abs((np.cumsum(y, axis=0) ** 2).sum(axis=0)) - 1E-4
            v_h2 = abs(((y[:-1] - y[1:]) ** 2).sum(axis=0)) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_h2[v_h2 < 0.0] = 0.0
            v = (v_h1 + v_h2) / 2

        elif self.problem_id == 11:
            f = y.sum(axis=0)
            v_g1 = np.prod(y, axis=0)
            v_h1 = abs(((y[:-1] - y[1:]) ** 2).sum(axis=0)) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_g1[v_g1 < 0.0] = 0.0
            v = (v_h1 + v_g1) / 2

        elif self.problem_id == 12:
            f = (y ** 2 - 10.0 * cos(2.0 * pi * y) + 10.0).sum(axis=0)
            v_g1 = 4.0 - (abs(y)).sum(axis=0)
            v_g2 = (y ** 2).sum(axis=0) - 4.0
            v_g1[v_g1 < 0.0] = 0.0
            v_g2[v_g2 < 0.0] = 0.0
            v = (v_g1 + v_g2) / 2

        elif self.problem_id == 13:
            f = (100.0 * (y[:-1] ** 2 - y[1:]) ** 2 + (y[:-1] - 1.0) ** 2).sum(axis=0)
            v_g1 = (y ** 2 - 10.0 * cos(2.0 * pi * y) + 10.0).sum(axis=0) - 100.0
            v_g2 = y.sum(axis=0) - 2.0 * self.dim
            v_g3 = 5.0 - y.sum(axis=0)
            v_g1[v_g1 < 0.0] = 0.0
            v_g2[v_g2 < 0.0] = 0.0
            v_g3[v_g3 < 0.0] = 0.0
            v = (v_g1 + v_g2) / 3

        elif self.problem_id == 14:
            f = -20.0 * exp(-0.2 * sqrt(1.0 / self.dim * (y**2).sum(axis=0))) + 20.0 \
                - exp(1.0 / self.dim * (cos(2.0 * pi * y)).sum(axis=0)) + e
            v_g1 = (y[1:]**2).sum(axis=0) + 1.0 - abs(y[0])
            v_h1 = abs((y**2).sum(axis=0) - 4.0) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_g1[v_g1 < 0.0] = 0.0
            v = (v_h1 + v_g1) / 2

        elif self.problem_id == 15:
            f = abs(y).max(axis=0)
            v_g1 = (y**2).sum(axis=0) - 100.0 * self.dim
            v_h1 = abs(cos(f) + sin(f)) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_g1[v_g1 < 0.0] = 0.0
            v = (v_h1 + v_g1) / 2

        elif self.problem_id == 16:
            f = (abs(y)).sum(axis=0)
            v_g1 = (y**2).sum(axis=0) - 100.0 * self.dim
            v_h1 = (cos(f) + sin(f))**2 - exp(cos(f) + sin(f) - 1.0 + exp(1.0)) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_g1[v_g1 < 0.0] = 0.0
            v = (v_h1 + v_g1) / 2

        elif self.problem_id == 17:
            arr = np.arange(1, self.dim + 1)
            f = np.prod((y.T / sqrt(1.0 + arr)).T, axis=0)
            data_sum = (y**2).sum(axis=0)
            y_ = np.tile(data_sum, (self.dim, 1))
            v_g1 = 1.0 - (np.sign(abs(y) - (y_ - y**2) - 1.0)).sum(axis=0)
            v_h1 = abs((y**2).sum(axis=0) - 4.0 * self.dim) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_g1[v_g1 < 0.0] = 0.0
            v = (v_h1 + v_g1) / 2

        elif self.problem_id == 18:
            v_g1 = 1 - (abs(y)).sum(axis=0)
            v_g2 = (y**2).sum(axis=0) - 100.0 * self.dim
            v_h1 = abs((100.0 * (y[:-1]**2 - y[1:])**2).sum(axis=0) + np.prod(sin((y - 1)**2 * pi))) - 1E-4
            np.where(abs(y) < 0.5, y, 0.5 * np.round(2.0 * y))
            f = (y ** 2 - 10.0 * cos(2.0 * pi * y) + 10.0).sum(axis=0)
            v_g1[v_g1 < 0.0] = 0.0
            v_g2[v_g2 < 0.0] = 0.0
            v_h1[v_h1 < 0.0] = 0.0
            v = (v_g1 + v_g2 + v_h1) / 3

        elif self.problem_id == 19:
            f = (abs(y)**0.5 + 2.0 * sin(y)**3).sum(axis=0)
            v_g1 = (-10.0 * exp(-0.2 * sqrt(y[:-1]**2 + y[1:]**2))).sum(axis=0) + (self.dim - 1.0) * 10.0 / exp(-5.0)
            v_g2 = (sin(2.0 * y)**2).sum(axis=0) - 0.5 * self.dim
            v_g1[v_g1 < 0.0] = 0.0
            v_g2[v_g2 < 0.0] = 0.0
            v = (v_g1 + v_g2) / 2

        elif self.problem_id == 20:
            f = (0.5 + (sin(sqrt(y[:-1]**2 + y[1:]**2))**2 - 0.5)
                 / (1.0 + 0.001 * sqrt(y[:-1]**2 + y[1:]**2))**2).sum(axis=0) \
                + 0.5 + (sin(sqrt(y[-1]**2 + y[1]**2))**2 - 0.5) / (1.0 + 0.001 * sqrt(y[-1]**2 + y[1]**2))**2
            v_g1 = cos(y.sum(axis=0))**2 - 0.25 * cos(y.sum(axis=0)) - 0.125
            v_g2 = exp(cos(y.sum(axis=0))) - exp(0.25)
            v_g1[v_g1 < 0.0] = 0.0
            v_g2[v_g2 < 0.0] = 0.0
            v = (v_g1 + v_g2) / 2

        elif self.problem_id == 21:
            z = self.rotate_func(y, self.matrix)
            f = (z**2 - 10.0 * cos(2.0 * pi * z) + 10.0).sum(axis=0)
            v_g1 = 4 - abs(z).sum(axis=0)
            v_g2 = (z**2).sum(axis=0) - 4.0
            v_g1[v_g1 < 0.0] = 0.0
            v_g2[v_g2 < 0.0] = 0.0
            v = (v_g1 + v_g2) / 2

        elif self.problem_id == 22:
            z = self.rotate_func(y, self.matrix)
            f = (100.0 * (z[:-1]**2 - z[1:])**2 + (z[:-1] - 1)**2).sum(axis=0)
            v_g1 = (z**2 - 10.0 * cos(2.0 * pi * z) + 10.0).sum(axis=0) - 100.0
            v_g2 = z.sum(axis=0) - 2.0 * self.dim
            v_g3 = 5.0 - z.sum(axis=0)
            v_g1[v_g1 < 0.0] = 0.0
            v_g2[v_g2 < 0.0] = 0.0
            v_g3[v_g3 < 0.0] = 0.0
            v = (v_g1 + v_g2 + v_g3) / 3

        elif self.problem_id == 23:
            z = self.rotate_func(y, self.matrix)
            f = -20.0 * exp(-0.2 * sqrt(1.0 / self.dim * (z**2).sum(axis=0))) + 20.0 \
                - exp(1.0 / self.dim * (cos(2.0 * pi * z)).sum(axis=0)) + e
            v_g1 = (z[1:]**2).sum(axis=0) + 1 - abs(z[0])
            v_h1 = abs((z**2).sum(axis=0) - 4.0) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_g1[v_g1 < 0.0] = 0.0
            v = (v_h1 + v_g1) / 2

        elif self.problem_id == 24:
            z = self.rotate_func(y, self.matrix)
            f = abs(z).max(axis=0)
            v_g1 = (z**2).sum(axis=0) - 100.0 * self.dim
            v_h1 = abs(cos(f) + sin(f)) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_g1[v_g1 < 0.0] = 0.0
            v = (v_h1 + v_g1) / 2

        elif self.problem_id == 25:
            z = self.rotate_func(y, self.matrix)
            f = (abs(z)).sum(axis=0)
            v_g1 = (z**2).sum(axis=0) - 100.0 * self.dim
            v_h1 = abs((cos(f) + sin(f))**2 - exp(cos(f) + sin(f)) - 1.0 + exp(1.0)) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_g1[v_g1 < 0.0] = 0.0
            v = (v_h1 + v_g1) / 2

        elif self.problem_id == 26:
            z = self.rotate_func(y, self.matrix)
            arr = np.arange(1, self.dim + 1)
            f = np.prod((z.T / sqrt(1.0 + arr)).T, axis=0)
            data_sum = (z ** 2).sum(axis=0)
            z_ = np.tile(data_sum, (self.dim, 1))
            v_g1 = 1.0 - (np.sign(abs(z) - (z_ - z**2) - 1.0)).sum(axis=0)
            v_h1 = abs((z**2).sum(axis=0) - 4.0 * self.dim) - 1E-4
            v_h1[v_h1 < 0.0] = 0.0
            v_g1[v_g1 < 0.0] = 0.0
            v = (v_h1 + v_g1) / 2

        elif self.problem_id == 27:
            z = self.rotate_func(y, self.matrix)
            v_g1 = 1 - (abs(z)).sum(axis=0)
            v_g2 = (z ** 2).sum(axis=0) - 100.0 * self.dim
            v_h1 = abs((100.0 * (z[:-1] ** 2 - z[1:]) ** 2).sum(axis=0) + np.prod(sin((z - 1)**2 * pi))) - 1E-4
            np.where(abs(z) < 0.5, z, 0.5 * np.round(2.0 * z))
            f = (z ** 2 - 10.0 * cos(2.0 * pi * z) + 10).sum(axis=0)
            v_g1[v_g1 < 0.0] = 0.0
            v_g2[v_g2 < 0.0] = 0.0
            v_h1[v_h1 < 0.0] = 0.0
            v = (v_g1 + v_g2 + v_h1) / 3

        elif self.problem_id == 28:
            z = self.rotate_func(y, self.matrix)
            f = (abs(z)**0.5 + 2.0 * sin(z)**3).sum(axis=0)
            v_g1 = (-10.0 * exp(-0.2 * sqrt(z[:-1] ** 2 + z[1:] ** 2))).sum(axis=0) \
                   + (self.dim - 1.0) * 10.0 / exp(-5.0)
            v_g2 = (sin(2.0 * z) ** 2).sum(axis=0) - 0.5 * self.dim
            v_g1[v_g1 < 0.0] = 0.0
            v_g2[v_g2 < 0.0] = 0.0
            v = (v_g1 + v_g2) / 2

        x[self.dim + 2, :] = f[:]
        x[self.dim + 3, :] = v[:]

    @staticmethod
    def get_lb_ub(problem_id):
        if problem_id in range(1, 4) or problem_id in [8] \
                or problem_id in range(10, 19) or problem_id in range(20, 28):
            lb = -100.0
            ub = 100.0
        elif problem_id in [4, 5, 9]:
            lb = -10.0
            ub = 10.0
        elif problem_id in [6]:
            lb = -20.0
            ub = 20.0
        elif problem_id in [7, 19, 28]:
            lb = -50.0
            ub = 50.0
        return lb, ub

