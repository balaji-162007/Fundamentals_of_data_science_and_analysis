import numpy as np
import scipy.stats as stats
group_1 = np.array([23, 45, 67, 32, 45, 34, 43, 45, 56, 42])
group_2 = np.array([45, 32, 23, 43, 46, 32, 21, 22, 43, 43])
group_3 = np.array([65, 78, 56, 67, 82, 73, 74, 65, 68, 74])
f_stat, p_value = stats.f_oneway(group_1, group_2, group_3)
print(f"F-statistic: {f_stat}")
print(f"P-value: {p_value}")
if p_value < 0.05:
    print("There is a significant difference between the group means.")
else:
    print("There is no significant difference between the group means.")
