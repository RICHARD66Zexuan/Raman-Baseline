from pybaselines import Baseline
import pandas as pd
import numpy as np
from setuptools.sandbox import save_path


#建立读取Excel数据的函数
def read_data_from_excel(file_path):
    # 使用 pandas 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 假设 Excel 文件中有两列数据：第一列为 x，第二列为 y
    x = df.iloc[:, 0].values  # 获取第一列数据作为 x
    y = df.iloc[:, 1].values  # 获取第二列数据作为 y

    return x, y

#模拟数据阶段#
############################################################################
# 模拟数据
#x = np.linspace(0, 100, 500)
#y = np.sin(x / 10) + np.random.normal(0, 0.1, 500) + 2  # 信号 + 噪声 + 背景

############################################################################

#基本参数设置
file_path = r"/Users/crystal-chenna/Desktop/Ninapython stud/pythonProject1/Baseline/Crystal violet data/10-5-3.xlsx"
output_file=r"/Users/crystal-chenna/Desktop/Ninapython stud/pythonProject1/Baseline/Test/5_3re.xlsx"

lam=2000
p=0.01
lam2=100
p2=0.004
start=530 #280
end=1700 #1700
diff_threshold = 50 #50 for 4，5，6，7，
#2e3 0.001
#1e2
#真实数据（青霉素）
# 读取 Excel 文件中的数据
x, y = read_data_from_excel(file_path)




#计算相邻点的差分
diff = np.abs(np.diff(y,n=3))
#diff_threshold = 50  # 设定变化率阈值
peaks = np.where(diff > diff_threshold)[0] + 1  # 找到异常峰位置
peaks = peaks[(peaks >= 1000) & (peaks <= 1800)] #限定异常峰的范围
print(peaks)

#补上缺失的转折点
# 找出缺失的值
full_sequence = np.arange(peaks.min(), peaks.max() + 1)
missing_values = np.setdiff1d(full_sequence, peaks)

# 检查连续性条件（前后至少连续 3 个）
valid_missing = []
for value in missing_values:
    # 找到与缺失值最近的前后元素
    prev_values = peaks[peaks < value]
    next_values = peaks[peaks > value]

    # 前连续性检查
    if value-prev_values[-1]<=2:
        prev_continuous = 'yes'
    else:
        prev_continuous = None

    # 后连续性检查
    if next_values[0]-value<=2:
        next_continuous = 'yes'
    else:
        next_continuous = None

    # 如果前后都有至少有3个连续数字之一，加入缺失值
    if (prev_continuous is not None) and (next_continuous is not None):
        valid_missing.append(value)

# 补全数组并排序
complete_array = np.sort(np.concatenate((peaks, valid_missing)))

print("缺失的值:", valid_missing)

peaks = complete_array

y_diff_cleaned = y.copy()
for peak in peaks:
    y_diff_cleaned[peak] = (y_diff_cleaned[peak - peaks.shape[0]] + y_diff_cleaned[peak + peaks.shape[0]]) / 2


x_peak=x[peaks]


#import matplotlib.pyplot as plt
#plt.plot(x, y, label="original", linestyle="--")
#plt.plot(x, y_diff_cleaned, label="Diff Cleaned", linestyle="--")
#plt.legend()
#plt.show()


#输出结果做扣除baseline处理
y=y_diff_cleaned


#仅对波长大于300的做baseline
x_filtered = [xi for xi in x if xi >= start and xi <= end]
y_filtered = [y[i] for i in range(len(x)) if x[i] >= start and x[i] <= end]

x=x_filtered
y=y_filtered


print(len(x))

import scipy.signal as signal
#采样频率
fs = 2810

# 设计低通滤波器
cutoff = 100  # 截止频率（低于这个频率的信号会被保留）
nyquist = 0.5 * fs  # 奈奎斯特频率
normal_cutoff = cutoff / nyquist  # 归一化截止频率

# 使用Scipy设计一个Butterworth低通滤波器
b, a = signal.butter(4, normal_cutoff, btype='low')

# 对含噪信号进行滤波
signal_filtered = signal.filtfilt(b, a, y)

#import matplotlib.pyplot as plt
#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # 2 行 1 列的子图
#ax1.plot(x, y, label="Original Data")
#ax2.plot(x, signal_filtered, label="Corrected Signal", color='g')
#plt.tight_layout()  # 调整布局以避免重叠
#plt.show()

y = signal_filtered

# 使用 AsLS 方法
baseline_fitter = Baseline()
baseline, corrected_signal = baseline_fitter.asls(y, lam=lam, p=p)

#second AsLS
y=y-baseline
baseline_fitter = Baseline()
baseline, corrected_signal = baseline_fitter.asls(y, lam=lam2, p=p2)

#baseline_fitter = Baseline()
#baseline, corrected_signal = baseline_fitter.asls(y, lam=lam, p=p)


import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # 2 行 1 列的子图

# 第一张图：原始数据和基线
ax1.plot(x, y, label="Original Data")
ax1.plot(x, baseline, label="Baseline", linestyle="--")
ax1.set_title("Original Data and Baseline")
ax1.set_xlabel("X-axis")
ax1.set_ylabel("Y-axis")
ax1.legend()


#处理出校正后的数据
y = np.array(y)
baseline = np.array(baseline)
corrected_data_array=y-baseline

# 第二张图：校正后的信号
ax2.plot(x, corrected_data_array, label="Corrected Signal", color='g')
ax2.set_title("Corrected Signal")
ax2.set_xlabel("X-axis")
ax2.set_ylabel("Y-axis")
ax2.legend()

# 显示图形
plt.tight_layout()  # 调整布局以避免重叠
plt.show()
#plt.savefig(save_path, dpi=300, bbox_inches="tight")


# 检查是否有任意一个元素在范围内
#is_any_in_range = np.all((x_peak >= 1180) & (x_peak <= 1200))
#print(f"是否有所有元素在范围 [{1180}, {1120}] 内: {is_any_in_range}")

import pandas as pd


df = pd.DataFrame({
    "column1": x,
    "column2": y - baseline  # 将 baseline 修正后的数据作为第二列
})

df.to_excel(output_file, index=False)
