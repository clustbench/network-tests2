import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
from scipy.fft import rfft, rfftfreq
import numpy as np
from matplotlib import pyplot as plt
import random
import numpy.linalg
import json
import gettext
import math

target_length = 500
target_src = 3
processors_range = (0, 4)
delays_range = (400, 600)

file_path = sys.argv[1]
cdf = nc.Dataset(file_path)

print(cdf.dimensions)
print(("Begin length:"), cdf['begin_mes_length'][:])
print(("Step:"), cdf['step_length'][:])
print(("End length:"), cdf['end_mes_length'][:])
print(("Proc amount:"), cdf['proc_num'][:])
proc_amount = cdf['proc_num'][:]
begin_length = cdf['begin_mes_length'][:]
step = cdf['step_length'][:]
data_with_fixed_length = cdf['data'][:][(target_length-begin_length)//step]
data_with_fixed_length = data_with_fixed_length[target_src]
dataset_values = {i: data_with_fixed_length[i][delays_range[0]:delays_range[1]] for i in range(processors_range[0], processors_range[1])}


min_delay = 1
for first_val in dataset_values:
    for cur_delay in dataset_values[first_val]:
        if cur_delay > 0.000000001 and cur_delay < min_delay:
            min_delay = cur_delay

print(min_delay)


for first_val in dataset_values:
    i = 0
    for cur_delay in dataset_values[first_val]:
        if cur_delay > min_delay*10:
            dataset_values[first_val][i] = min_delay*10
        i += 1

df = pd.DataFrame(dataset_values, [i for i in range(delays_range[0], delays_range[1])])
sns.set(font_scale = 2)
sns.heatmap(df, cmap="YlGnBu")
plt.show()
