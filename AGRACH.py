import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt




# 定义AGARCH模型
class AGARCH:
    def __init__(self, R, VIX, rf, spx):
        self.R = R  # 收益率
        self.VIX = VIX  # VIX指数
        self.rf = rf  # 无风险利率
        self.spx = spx
        self.N = len(R)

    def compute_hbar_and_B(self, params):
        """
        计算hbar和B参数
        hbar = omega / (1 - beta - alpha * gamma^2)
        B = beta + alpha * gamma^2
        """
        omega, alpha, beta, gamma, lambd = params
        B = beta + alpha * (gamma + lambd) ** 2
        if B >= 1:
            return np.inf, np.inf  # 使得该参数组合不可行
        hbar = (omega + alpha) / (1 - B)
        return hbar, B

    def vix_model(self, h_next, hbar, B):
        """
        根据公式 (35) 计算模型的VIX值
        VIXt = 100 * sqrt(252 / 22) * sqrt(Ht)
        Ht = a + b * h_next
        其中，a = hbar / (1 - B)
               b = (1 - B^22) / (1 - B)
        """
        AF = 100 * np.sqrt(252 / 22)
        a = hbar / (1 - B)
        b = (1 - B ** 22) / (1 - B)
        Ht = a + b * h_next
        return AF * np.sqrt(Ht)

    def log_likelihood(self, params):
        """
        计算联合对数似然函数
        """
        omega, alpha, beta, gamma, lambd = params
        hbar, B = self.compute_hbar_and_B(params)

        if np.isinf(hbar) or np.isinf(B):
            return 1e10  # 返回一个很大的负对数似然值

        h_0 = np.var(self.spx)
        # 初始化h_t
        h = np.full(self.N + 1, h_0)  # h[0] 是初始条件
        for t in range(self.N):
            varepsilon_t = (self.R[t] - self.rf[t] - lambd * h[t] + 0.5 * h[t]) / np.sqrt(h[t])
            h[t + 1] = omega + beta * h[t] + alpha * (varepsilon_t - gamma * np.sqrt(h[t])) ** 2

        # 计算收益率对数似然
        logL_R = -0.5 * self.N * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(h[1:self.N + 1]) + ((self.R - self.rf - lambd * h[1:self.N + 1] + 0.5 * h[1:self.N + 1]) ** 2) / h[
                                                                                                                    1:self.N + 1])

        # 计算VIX模型值
        # 根据公式 (35): log VIX = log(AF) + 0.5 * log(Ht)
        AF = 100 * np.sqrt(252 / 22)

        b = (1 - B ** 22) / (1 - B)
        a = hbar / (22 - b)
        Ht = a + b * h[1:self.N + 1]

        # 对数VIX
        log_VIX = np.log(AF) + 0.5 * np.log(Ht)

        # 计算测量误差
        u = self.VIX - np.exp(log_VIX)
        sigma_sq = np.var(u)

        # 计算VIX对数似然
        logL_VIX = -0.5 * self.N * np.log(2 * np.pi) - 0.5 * self.N * np.log(sigma_sq) - 0.5 * np.sum(
            (u ** 2) / sigma_sq)

        # 联合对数似然
        logL = logL_R + logL_VIX
        return -logL  # 由于我们使用最小化函数，取负

    def estimate_parameters(self, initial_guess):
        """
        使用最大似然估计（MLE）来估计模型参数
        """

        bounds = [(1e-6, None), (1e-6, 1 - 1e-6), (1e-6, 1 - 1e-6), (0, 10), (0, 10)]
        result = minimize(self.log_likelihood, initial_guess, method='L-BFGS-B', bounds=bounds)
        if result.success:
            estimated_params = result.x
            print("估计的参数：", estimated_params)
            return estimated_params
        else:
            raise ValueError("参数估计未收敛")

    def compute_moments(self, params):
        """
        计算对数VIX的前四阶原点矩
        """
        omega, alpha, beta, gamma, lambd = params
        hbar, B = self.compute_hbar_and_B(params)

        if np.isinf(hbar) or np.isinf(B):
            return None, None, None, None

        h_0 = np.var(self.spx)
        # 初始化h_t
        h = np.full(self.N + 1, h_0)  # h[0] 是初始条件
        for t in range(self.N):
            varepsilon_t = (self.R[t] - self.rf[t] - lambd * h[t] + 0.5 * h[t]) / np.sqrt(h[t])
            h[t + 1] = omega + beta * h[t] + alpha * (varepsilon_t - gamma * np.sqrt(h[t])) ** 2

        # 计算Ht
        AF = 100 * np.sqrt(252 / 22)
        a = hbar / (1 - B)
        b = (1 - B ** 22) / (1 - B)
        Ht = a + b * h[1:self.N + 1]

        # 对数VIX
        log_VIX = np.log(AF) + 0.5 * np.log(Ht)

        # 计算矩
        mu = np.mean(log_VIX)
        sigma_sq = np.var(log_VIX)
        sigma = np.sqrt(sigma_sq)
        tau3 = (np.mean((log_VIX - mu) ** 3)) / sigma ** 3
        tau4 = (np.mean((log_VIX - mu) ** 4)) / sigma ** 4 - 3

        return mu, sigma_sq, tau3, tau4