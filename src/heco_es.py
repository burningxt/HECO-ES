from benchmark.cec2017.cec2017 import Cec2017
from scipy.stats import rankdata
import numpy as np
import pandas as pd
import random


class MAES(Cec2017):
    def __init__(self, problem_id, dim, o_shift, matrix, matrix1, matrix2):
        super().__init__(problem_id, dim, o_shift, matrix, matrix1, matrix2)
        self.problem_id = problem_id
        self.dim = dim
        self.λ = 4 * dim
        self.ϵ = 1E-4
        self.w_0 = 0.8
        self.θ = 0.2

    def keep_range(self, x_, d_, z_, mean_, mm_, lb, ub, σ):
        mask1 = np.where(x_[:self.dim, :] < lb)
        mask2 = np.where(x_[:self.dim, :] > ub)
        x_[mask1] = lb + (lb - x_[mask1] - np.floor((lb - x_[mask1]) / (ub - lb)) * (ub - lb))
        x_[mask2] = ub - (x_[mask2] - ub - np.floor((x_[mask2] - ub) / (ub - lb)) * (ub - lb))
        if np.asarray(mask1).shape[1] != 0 and np.asarray(mask2).shape[1] != 0:
            pim = np.linalg.pinv(mm_)
            d_[:, :] = (x_[:self.dim, :].T - mean_[:, 0]).T / σ
            z_[:, :] = pim.dot(d_)

    def equ(self, pop):
        feasible_solutions = pop[:, pop[self.dim + 3, :] == 0.0]
        f_min = np.amin(pop[self.dim + 2, :])
        fea_size = feasible_solutions.shape[1]
        if fea_size:
            f_feasible = feasible_solutions[self.dim + 2, random.randint(0, fea_size - 1)]
            if f_min < 0.0:
                pop[self.dim + 1, :] = pop[self.dim + 2, :] + abs(f_min) + f_feasible
            else:
                pop[self.dim + 1, :] = pop[self.dim + 2, :] + f_feasible
        else:
            if f_min < 0.0:
                pop[self.dim + 1, :] = pop[self.dim + 2, :] + abs(f_min)
            else:
                pop[self.dim + 1, :] = pop[self.dim + 2, :]

    def normalization(self, x, weight):
        arg_equ = rankdata(x[self.dim + 1, :])
        arg_vio = rankdata(x[self.dim + 3, :])
        pop_size = x.shape[1]
        x[self.dim + 7, :] = arg_equ[:] / pop_size
        x[self.dim + 8, :] = arg_vio[:] / pop_size
        x[self.dim, :] = weight[0, :] * x[self.dim + 7, :] + weight[1, :] * x[self.dim + 8, :]

    @staticmethod
    def converge_state(old_best, current_best):
        if old_best != 0.0:
            δ = (old_best - current_best) / abs(old_best)
        elif old_best - current_best != 0.0:
            δ = 1.0
        else:
            δ = 0.0
        return δ

    def ma_es(self):
        σ = 1.0
        p_σ = np.zeros((self.dim, 1))
        μ = int(np.floor(self.λ / 3))
        arr_μ = np.arange(1, μ + 1)
        w = np.zeros((μ, 1))
        w[:, 0] = (np.log(μ + 0.5) - np.log(arr_μ)) / np.sum((np.log(μ + 0.5) - np.log(arr_μ)))
        weight = np.zeros((2, self.λ))
        μ_eff = 1.0 / np.sum(w ** 2)
        parent_z = np.zeros((self.dim, 1))
        parent_d = np.zeros((self.dim, 1))
        c_σ = (μ_eff + 2.0) / (self.dim + μ_eff + 5.0)
        c_1 = 2.0 / ((self.dim + 1.3)**2 + μ_eff)
        c_μ = min(1.0 - c_1, 2.0 * (μ_eff - 2.0 + 1.0 / μ_eff) / ((self.dim + 2.0)**2 + μ_eff))
        lb, ub = self.get_lb_ub(self.problem_id)
        σ_max = 100.0
        x = lb + np.random.random((self.dim + 10, self.λ)) * (ub - lb)
        x[self.dim:, :] = 0.0
        mm = np.identity(self.dim)
        identity = np.identity(self.dim)
        self.benchmark(x)
        self.equ(x)
        mean_w = np.array([self.w_0, 1.0 - self.w_0])
        weight[0, :] = np.random.uniform(mean_w[0] - 0.5 * self.θ, mean_w[0] + 0.5 * self.θ)
        weight[1, :] = 1.0 - weight[0, :]
        self.normalization(x, weight)
        all_time_best = np.zeros((self.dim + 10, 1))
        all_time_best[:, 0] = x[:, np.lexsort((x[self.dim + 2, :], x[self.dim + 3, :]))[0]]
        current_best = np.zeros((self.dim + 10, 1))
        old_best = np.zeros((self.dim + 10, 1))
        old_best[:, 0] = all_time_best[:, 0]
        mean = np.zeros((self.dim, 1))
        mean[:, 0] = np.sum(x[:self.dim, :μ], axis=1) / μ
        fes_data = []
        weight_e_data = []
        weight_v_data = []
        f_data = []
        e_data = []
        v_data = []
        delta_e = []
        delta_v = []
        case = []
        fes_max = 20000 * self.dim
        fes = self.λ
        while fes < fes_max:
            z = np.random.randn(self.dim, self.λ)
            d = mm.dot(z)
            x[:self.dim, :] = np.tile(mean, (1, self.λ)) + σ * d
            self.keep_range(x, d, z, mean, mm, lb, ub, σ)
            self.benchmark(x)
            weight[0, :] = np.random.uniform(mean_w[0] - 0.5 * self.θ, mean_w[0] + 0.5 * self.θ)
            weight[1, :] = 1.0 - weight[0, :]
            self.equ(x)
            self.normalization(x, weight)
            arg_λ = x[self.dim, :].argsort()
            arg_μ = arg_λ[:μ]
            x[:, :] = x[:, arg_λ]
            parent_z[:, 0] = np.sum(z[:, arg_μ].dot(w), axis=1)
            parent_d[:, 0] = np.sum(d[:, arg_μ].dot(w), axis=1)
            mean[:] = mean + σ * parent_d

            p_σ[:] = (1.0 - c_σ) * p_σ + (μ_eff * c_σ * (2.0 - c_σ))**0.5 * parent_z
            mm[:] = (1.0 - 0.5 * c_1 - 0.5 * c_μ) * mm + 0.5 * c_1 * (mm.dot(p_σ)).dot(p_σ.T)
            mm[:] = mm + mm.dot((0.5 * c_μ * z[:, arg_μ] * w.T).dot(z[:, arg_μ].T))
            σ = min(σ * np.exp(c_σ / 2.0 * ((p_σ[:, 0]**2).sum() / self.dim - 1.0)), σ_max)

            current_best[:, 0] = x[:, np.lexsort((x[self.dim + 2, :], x[self.dim + 3, :]))[0]]
            if current_best[self.dim + 3, 0] < all_time_best[self.dim + 3, 0]:
                all_time_best[:, 0] = current_best[:, 0]
            elif current_best[self.dim + 3, 0] == all_time_best[self.dim + 3, 0] and \
                    (current_best[self.dim + 2, 0] < all_time_best[self.dim + 2, 0]):
                all_time_best[:, 0] = current_best[:, 0]
            δ_equ = self.converge_state(old_best[self.dim + 1], current_best[self.dim + 1])
            δ_vio = self.converge_state(old_best[self.dim + 3], current_best[self.dim + 3])
            δ_equ = min(0.1, δ_equ)
            δ_equ = max(-0.1, δ_equ)
            δ_vio = min(0.1, δ_vio)
            δ_vio = max(-0.1, δ_vio)
            if fes % (self.λ * self.dim) == 0.0:
                # δ_equ = self.converge_state(old_best[self.dim + 1], current_best[self.dim + 1])
                # δ_vio = self.converge_state(old_best[self.dim + 3], current_best[self.dim + 3])
                if δ_equ > self.ϵ:
                    if -self.ϵ < δ_vio < self.ϵ:
                        if current_best[self.dim + 3] > self.ϵ:
                            mean_w[0] = 1.0 - self.w_0 + (self.w_0 - (1.0 - self.w_0)) * random.random()
                            σ = σ_max
                            mm[:] = identity[:]
                            case.append(1)
                        else:
                            case.append(0)
                    else:
                        case.append(0)

                elif -self.ϵ < δ_equ <= self.ϵ:
                    if -self.ϵ < δ_vio <= self.ϵ:
                        if current_best[self.dim + 3] > self.ϵ:
                            mean_w[0] = 1.0 - self.w_0 + (self.w_0 - (1.0 - self.w_0)) * random.random()
                            σ = σ_max
                            mm[:] = identity[:]
                            case.append(2)
                        elif current_best[self.dim + 3] == 0.0:
                            mean_w[0] = self.w_0
                            # mean_w[0] = 1.0 - self.w_0 + (self.w_0 - (1.0 - self.w_0)) * random.random()
                            σ = σ_max
                            mm[:] = identity[:]
                            case.append(3)
                        else:
                            case.append(0)
                    elif δ_vio <= -self.ϵ:
                        mean_w[0] = max(1.0 - self.w_0, mean_w[0] - self.θ)
                        σ = σ_max
                        mm[:] = identity[:]
                        case.append(4)
                    else:
                        case.append(0)
                else:
                    if δ_vio <= -self.ϵ:
                        mean_w[0] = max(1.0 - self.w_0, mean_w[0] - self.θ)
                        # σ = σ_max
                        # mm[:] = identity[:]
                        case.append(5)
                    elif δ_vio > self.ϵ:
                        mean_w[0] = max(1.0 - self.w_0, mean_w[0] - self.θ)
                        case.append(6)
                    else:
                        case.append(0)
                old_best[:, 0] = current_best[:, 0]
                weight[0, :] = np.random.uniform(mean_w[0] - 0.5 * self.θ, mean_w[0] + 0.5 * self.θ)
                weight[1, :] = 1.0 - weight[0, :]
            # elif fes % 100 == 0:
            #     case.append(0)


            print(fes, all_time_best[self.dim + 2, 0], all_time_best[self.dim + 3, 0], current_best[self.dim + 2],
                  current_best[self.dim + 3], σ, mean_w[0])

            if fes % (self.λ * self.dim) == 0:
                fes_data.append(fes)
                weight_e_data.append(np.mean(weight[0, :]))
                weight_v_data.append(np.mean(weight[1, :]))
                f_data.append(min(15, current_best[self.dim + 2, 0]))
                e_data.append(min(15, current_best[self.dim + 1, 0]))
                v_data.append(min(10, current_best[self.dim + 3, 0]))
                delta_e.append(δ_equ)
                delta_v.append(δ_vio)
                if fes % (self.λ * self.dim) != 0:
                    case.append(0)

            fes += self.λ

        df_data = pd.DataFrame(
            {
                "fes": fes_data,
                "weight_e": weight_e_data,
                "weight_v": weight_v_data,
                "best_f": f_data,
                "best_e": e_data,
                "best_v": v_data,
                "δ_e": delta_e,
                "δ_v": delta_v,
                "case": case
            }
        )
        return all_time_best[self.dim + 2, 0], all_time_best[self.dim + 3, 0], int(all_time_best[self.dim + 4, 0]),\
               int(all_time_best[self.dim + 5, 0]), int(all_time_best[self.dim + 6, 0]), df_data

