import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa import stattools
import statsmodels.api as sm
import matplotlib

# demo dataset
filename = 'bitcoin_price_history.csv'
df = pd.read_csv(filename)

df['Date'] = pd.to_datetime(df['Date'])         # set the 'Date' column to be datatime type
df = df.sort_values(by="Date", ascending=True)  # sort the dataset by Date


## 平稳性检验
# Method 1: time series plot
fig, ax = plt.subplots()
fig.set_size_inches(9, 3.5)
ax.plot(df['Date'], df['Close'])
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(90))
plt.xticks(rotation=90)
ax.tick_params(labelsize=7)
plt.show()


# Method 2: calculate and plot ACF and PACF
n_lag = 100 ###lag
acf = stattools.acf(df['Close'], nlags=n_lag) #Autocorrelation Coefficient
pacf = stattools.pacf(df['Close'], nlags=n_lag)
print('Autocorrelation Coefficient (ACF): \n{}'.format(acf))
print('Partial Autocorrelation Coefficient (PACF): \n{}'.format(pacf))

sm.graphics.tsa.plot_acf(df['Close'], lags=n_lag)
plt.show()  # 阴影部分是置信区间。默认情况下，置信区间被设置为95%。

sm.graphics.tsa.plot_pacf(df['Close'], lags=n_lag)
plt.show()

# 单位根检验（这里采用ADF检验，分别用两种方法进行，第二种的输出效果较好)
from statsmodels.stats.diagnostic import unitroot_adf
adf_method1 = unitroot_adf(df['Close'])
print('ADF method 1: \n{}'.format(adf_method1))

from arch.unitroot import ADF
adf_method2 = ADF(df['Close'])
print('ADF method 2: \n{}'.format(adf_method2))