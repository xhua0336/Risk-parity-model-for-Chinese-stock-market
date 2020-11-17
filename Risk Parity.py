#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import time

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('RP.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df.head(10)


# In[3]:





# In[4]:


def get_return_df(df):
    returns_df = (df-df.shift(1))/df.shift(1) # 简单收益率
    # returns_df = np.log(df/df.shift(1)) # 对数收益率
    returns_df.dropna(axis='index', inplace=True) # 删除空数据
    return returns_df
index_return = get_return_df(df)


# In[5]:


def get_train_set(change_time,df):
    """返回训练样本数据"""
# change_time: 调仓时间
    df = df.loc[df.index<change_time] 
    df = df.iloc[-240:] #每个调仓前240个交易日
    return df


# ## Risk Parity Model

# In[6]:


from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_gaussian_quantiles


# In[8]:


def calculate_cov_matrix(df):
    """计算协方差矩阵"""
    df = df/df.iloc[0]*100 # 统一缩放到100为基点
    returns_df = (df-df.shift(1))/df.shift(1) # 简单收益率
    returns_df.dropna(axis='index', inplace=True) # 删除空数据
    one_cov_matrix = returns_df.cov()*250
    return np.matrix(one_cov_matrix)


# In[9]:


# 标准风险平价下的风险贡献
def calculate_risk_contribution(weight,one_cov_matrix):
    weight=np.matrix(weight) 
    sigma=np.sqrt(weight*one_cov_matrix*weight.T)
    # 边际风险贡献 Marginal Risk Contribution (MRC)
    MRC=one_cov_matrix*weight.T/sigma
    # 风险贡献 Risk Contribution (RC)
    RC=np.multiply(MRC,weight.T)
    return RC


# In[10]:


import scipy.optimize as sco
import scipy.interpolate as sci

# 定义优化问题的目标函数，即最小化资产之间的风险贡献差
def naive_risk_parity(weight,parameters): 
    # weight: 待求解的资产权重,
    # parameters: 参数列表 
    # parameters[0]: 协方差矩阵
    # parameters[1]: 风险平价下的目标风险贡献度向量

    one_cov_matrix=parameters[0]
    RC_target_ratio=parameters[1] 
    # RC_target为风险平价下的目标风险贡献，一旦参数传递以后，RC_target就是一个常数，不随迭代而改变
    sigma_portfolio=np.sqrt(weight*one_cov_matrix*np.matrix(weight).T) # 组合波动率
    RC_target=np.asmatrix(np.multiply(sigma_portfolio,RC_target_ratio))    # 目标风险贡献
    # RC_real是 每次迭代以后最新的真实风险贡献，随迭代而改变
    RC_real=calculate_risk_contribution(weight,one_cov_matrix)
    sum_squared_error= sum(np.square(RC_real-RC_target.T))[0,0] 
    return sum_squared_error


# In[11]:


# 根据资产预期目标风险贡献度来计算各资产的权重
def calculate_portfolio_weight(one_cov_matrix,risk_budget_objective):
    '''
约束条件的类型只有'eq'和'ineq'两种
eq表示约束方程的返回结果为0
ineq表示约束方程的返回结果为非负数
'''
    num = df.shape[1]
    x0 = np.array([1.0 / num for _ in range(num)]) # 初始资产权重
    bounds = tuple((0, 1) for _ in range(num))     # 取值范围(0,1)

    cons_1 = ({'type': 'eq', 'fun': lambda x: sum(x) - 1},)   #权重和为1
    RC_set_ratio = np.array([1.0 / num for _ in range(num)])   #风险平价下每个资产的目标风险贡献度相等
    optv = sco.minimize(risk_budget_objective, x0, args=[one_cov_matrix,RC_set_ratio], method='SLSQP', bounds=bounds, constraints=cons_1)
    return optv.x


# In[12]:


def get_weight_matrix(df, method = None):
    """返回资产权重矩阵"""
    period_type = 'D'
    df_weight= df
    df_weight = df_weight[df_weight.index>='2005-01-04']
    
    for i in range(len(df_weight.index)):
        one_cov_matrix = calculate_cov_matrix(df)
        df_weight.iloc[i] = calculate_portfolio_weight(one_cov_matrix,naive_risk_parity)
 
    return df_weight


# In[13]:


df_weight_rp = get_weight_matrix(df,method=None)


# In[14]:


df_weight_rp


# In[23]:


#daily return
pct_daily = df.pct_change()
pct_daily = pct_daily.fillna(0)

#Cal RP return
rp_return = pd.DataFrame()
rp_return = np.multiply(df_weight_rp, pct_daily)
sum_column = rp_return["stock"] + rp_return["debt"]
rp_return["return"] = sum_column
rp_return = rp_return["return"]


# In[24]:


rp_return.to_frame(name='return')


# In[25]:


#计算净值
a = np.zeros(shape=(3740,1))
jinzhi = pd.DataFrame(a,columns=['jinzhi'])
jinzhi.loc[0:0] = 1


# In[26]:


i = 1
while i <= 3739:
    jinzhi.iloc[i] = jinzhi.iloc[i-1]*(1+rp_return.iloc[i])
    i+=1


# In[27]:


jinzhi


# In[28]:


# 计算最大回撤
def max_draw_down(ret): 
    ret = np.array(ret)
    index_j = np.argmax(np.maximum.accumulate(ret) - ret)  # 结束位置
    index_i = np.argmax(ret[:index_j])  # 开始位置
    dd = ret[index_i] - ret[index_j]  # 最大回撤
    dd = dd/ret[index_i]
    return dd,index_i,index_j


# In[29]:


def get_eval_indicator(ret):
    """各种评价指标"""
    eval_indicator = pd.Series(index=['年化收益率','年化波动率','最大回撤','sharpe比率','Calmar比率'])
    return_df = get_return_df(ret)
    #年化收益率  
    annual_ret = (ret.iloc[-1]/ret.iloc[0])**(250/ret.shape[0])-1
    eval_indicator['年化收益率'] = annual_ret
    #annual_ret = np.power(1+return_df.mean(), 250)-1 # 几何年化收益
    #年化波动率
    sigma = rp_return.std() * (250**0.5)
    eval_indicator['年化波动率'] = sigma
    # 最大回撤
    dd, dd_index_start, dd_index_end = max_draw_down(ret)
    eval_indicator['最大回撤'] = dd
    #夏普比率  无风险利率是3%
    bench_annual_ret = 0.03
    sharpe = (annual_ret-bench_annual_ret)/sigma
    eval_indicator['sharpe比率'] = sharpe
    #Calmar比率=年化收益率/最大历史回撤
    calmar = (annual_ret-bench_annual_ret)/dd
    eval_indicator['Calmar比率'] = calmar
    return eval_indicator


# In[30]:


get_eval_indicator(jinzhi).round(4)


# In[ ]:




