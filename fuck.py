import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt

from AGRACH import AGARCH

# 读取数据
# 加载SPX收益率数据
spx_data = pd.read_csv('SPX_data.csv', parse_dates=['Date'])
# spx_df.set_index('Date', inplace=True)

# 加载VIX数据
vix_data = pd.read_csv('VIX_data.csv', parse_dates=['Date'])
# vix_df.set_index('Date', inplace=True)

# 读取利率数据 - 处理科学计数法和缺失值
daily_rates = pd.read_csv('daily_continuous_rates.csv',
                          converters={'Rate': lambda x: float(x) if x != '' else 0})
daily_rates['Date'] = pd.to_datetime(daily_rates['Date'])
daily_rates.set_index('Date', inplace=True)

# 将缺失值填充为0
daily_rates['Rate'] = daily_rates['Rate'].fillna(0)

# 确保所有科学计数法的数值都被正确转换为浮点数
daily_rates['Rate'] = daily_rates['Rate'].apply(lambda x: float(x))

rf_data = daily_rates

# 合并数据基于日期
data = pd.merge(spx_data, vix_data, on='Date', how='inner')
data = pd.merge(data, rf_data, on='Date', how='inner')
data = data.rename(columns={'Value_y':'VIX','Value_x':'SPX_Close'})

# 重命名列（根据实际情况调整）
data.rename(columns={
    'Close': 'SPX_Close',
    'VIX': 'VIX',
    'Rate': 'RF'
}, inplace=True)
print(data.head())
# 按日期排序
data.sort_values('Date', inplace=True)

# 计算SPX对数收益率
data['SPX_Return'] = np.log(data['SPX_Close'] / data['SPX_Close'].shift(1))

# 删除缺失值
data.dropna(inplace=True)

# 重置索引
data.reset_index(drop=True, inplace=True)

# 查看数据
print(data.head())


# 定义AGARCH模型



# 定义定价函数
def call_option_payoff(x, mu, sigma, K):
    return np.maximum(np.exp(mu + x * sigma) - K, 0)


def C1(mu, sigma, K, r, T_minus_t, tau3, tau4):
    """
    标准正态分布下的定价公式
    """

    def integrand(x):
        payoff = call_option_payoff(x, mu, np.sqrt(sigma), K)
        normal_pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / 2)
        correction = 1 + (x ** 3 - 3 * x) / 6 * tau3 + (x ** 4 - 6 * x ** 2 + 3) / 24 * (tau4 - 3)
        return payoff * normal_pdf * correction

    price, _ = quad(integrand, -10, 10)  # 取-10到10作为积分区间的近似
    return np.exp(-r * T_minus_t) * price


def C2(mu, sigma, K, r, T_minus_t, tau3, tau4):
    """
    Logistic分布下的定价公式
    """

    def integrand(x):
        payoff = call_option_payoff(x, mu, np.sqrt(sigma), K)
        logistic_pdf = (np.pi / (np.sqrt(3) * (np.exp(np.pi * x / np.sqrt(3)) + 2 + np.exp(-np.pi * x / np.sqrt(3)))))
        correction = 1 + 175 / 3888 * (x ** 3 - 21 / 5 * x) * tau3 + 1225 / 331776 * (
                    x ** 4 - 78 / 7 * x ** 2 + 243 / 35) * (tau4 - 147 / 35)
        return payoff * logistic_pdf * correction

    price, _ = quad(integrand, -100, 100)  # 取-10到10作为积分区间的近似
    return np.exp(-r * T_minus_t) * price

# 提取必要的数据
R = data['SPX_Return'].values
VIX = data['VIX'].values
rf = data['RF'].values  # 假设RF为每日无风险利率
spx = data['SPX_Close'].values.astype(float)
# 初始化AGARCH模型
agarch_model = AGARCH(R, VIX, rf, spx)

# 初始参数猜测
initial_guess = [0.1, 0.1, 0.8, 0.1, 0.1]  # [omega, alpha, beta, gamma, lambda]

# 估计参数
estimated_params = agarch_model.estimate_parameters(initial_guess)
omega, alpha, beta, gamma, lambd = estimated_params

# 计算条件矩
mu, sigma_sq, tau3, tau4 = agarch_model.compute_moments(estimated_params)
print(f"mu: {mu}, sigma_sq: {sigma_sq}, tau3: {tau3}, tau4: {tau4}")

# # 定义期权参数
# K = 20  # 执行价格，根据实际需求调整
# T_minus_t = 30 / 252  # 例如一个月到期，252个交易日
#
# # 计算C1和C2
# C1_price = C1(mu, sigma_sq, K, np.mean(rf), T_minus_t, tau3, tau4)
# C2_price = C2(mu, sigma_sq, K, np.mean(rf), T_minus_t, tau3, tau4)
#
# print(f"标准正态分布下的VIX期权价格: {C1_price}")
# print(f"Logistic分布下的VIX期权价格: {C2_price}")
