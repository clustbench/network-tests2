import netCDF4 as nc
import sys
from scipy.fft import rfft, rfftfreq
import numpy as np
from matplotlib import pyplot as plt
import gettext

def draw_stat(cur_1d):
    plt.hist(cur_1d, color = 'blue', edgecolor = 'black', bins = 100)
    plt.title('Histogram of delays')
    plt.xlabel('Delay (ms)')
    plt.ylabel('amount of iterations')
    plt.show()


file_path = ("/home/volch/3_kurs/asvk_sc/results/" + sys.argv[1])
cdf = nc.Dataset(file_path)

print(type(cdf))
print('\n\n')
print(_("Begin length:"), cdf['begin_mes_length'][:])
print(_("Step:"), cdf['step_length'][:])
print(_("End length:"), cdf['end_mes_length'][:])


while (params := input(_("Введите длину сообщения, для которой надо создать трехмерный файл: "))) != 'quit':
    params = params.split()
    print(params)
    mes_length = int(params[0])
    mes_length //= cdf['step_length'][:]
    n_src = int(params[1])
    n_dest = int(params[2])

    data = cdf['data'][:]

    try:
        cur_1d = data[mes_length][n_src][n_dest]
        draw_stat(cur_1d)
    except IndexError:
        print(_("your length is not valid"))
    
##    else:
##        print(type(cur_3d), cur_3d)
##        ##cur_spektrum = spektrum(cur_3d,1,2,20,120)
##        spektrum_ncdf(cur_3d, int(input("Введите длину окна для подсчета дискретного спектра: ")), mes_length)
