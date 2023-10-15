import netCDF4 as nc
import sys
from scipy.fft import rfft, rfftfreq
import numpy as np
from matplotlib import pyplot as plt
import gettext


def spektrum(cur_3d, src, dest, pos_begin, length):
    SAMPLE_RATE = length
    DURATION = 1

    normalized_tone = np.float64(cur_3d[src][dest][pos_begin:pos_begin+length])
    # число точек в normalized_tone
    N = SAMPLE_RATE * DURATION

    yf = np.abs(rfft(normalized_tone))
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    print(xf, yf)
    input()
    
    plt.plot(xf, yf)
    plt.show()

    border_const = 1000
    filtered_xf = []
    filtered_yf = []
    cur_border = min(yf) * border_const
    print(yf, cur_border)
    for i in range(len(yf)):
        if yf[i] <= cur_border:
            filtered_yf.append(yf[i])
            filtered_xf.append(xf[i])
    plt.plot(filtered_xf, filtered_yf)
    plt.show()


def spektrum_ncdf(cur_3d, win_length, mes_length):
    proc_range = len(cur_3d)
    measure_amount = len(cur_3d[0][0]) - win_length

    data = [[[None for i2 in range(measure_amount)] for i3 in range(proc_range)] for i4 in range(proc_range)]
    for src in range(proc_range):
        for dest in range(proc_range):
            for pos_begin in range(measure_amount):
                SAMPLE_RATE = win_length
                DURATION = 1
                normalized_tone = np.float64(cur_3d[src][dest][pos_begin:(pos_begin + win_length)])

                N = SAMPLE_RATE * DURATION
                yf = rfft(normalized_tone)
                print(":YF:", yf)
                yf = np.abs(yf)
                print(":abs(YF):", yf)
                input()
                xf = rfftfreq(N, 1 / SAMPLE_RATE)

                data[src][dest][pos_begin] = yf

    spek_size = len(data[0][0][0])
    new_netcdf = nc.Dataset(f"spectrum_{mes_length}_{win_length}.nc", mode = 'w', format = 'NETCDF3_64BIT_OFFSET')
    src_dim = new_netcdf.createDimension('src', size = proc_range)
    dest_dim = new_netcdf.createDimension('dest', size = proc_range)
    pos_dim = new_netcdf.createDimension('pos', size = measure_amount)
    spek_dim = new_netcdf.createDimension('spek', size = spek_size)
    ncdf_data = new_netcdf.createVariable('data', 'f8', dimensions = (src_dim,dest_dim,pos_dim,spek_dim))
    ncdf_data[:] = data


def make_netcdf(cdf, cur_3d, cur_size):
    tmp = cdf['data']
    print(tmp.dimensions, tmp.shape)
    attrs = {}
    for i in ["group()", "name", "datatype", "dimensions", "compression", "zlib", "complevel", "shuffle", "szip_coding", "szip_pixels_per_block", "blosc_shuffle", "fletcher32", "contiguous", "chunksizes", "endian()", "least_significant_digit", "fill_value", "chunk_cache"]:
        try:
            cur = eval("tmp."+i)
        except:
            continue
        else:
            attrs[i] = cur
    for i in attrs:
        print(i, attrs[i], type(attrs[i]), '\n')

    new_netcdf = nc.Dataset(f"fixed_length_{cur_size}.nc", mode = 'w', format = 'NETCDF3_64BIT_OFFSET')
    
    n = new_netcdf.createDimension('n', size = tmp.shape[3])
    x = new_netcdf.createDimension('x', size = tmp.shape[2])
    y = new_netcdf.createDimension('y', size = tmp.shape[1])
    s = new_netcdf.createDimension('strings', size = cdf.dimensions['strings'].size)

    print(cdf['proc_num'].getValue())
    proc_num = new_netcdf.createVariable('proc_num', 'i4')
    test_type = new_netcdf.createVariable('test_type', 'S1', dimensions = (s))
    data_type = new_netcdf.createVariable('data_type', 'i4')
    begin_mes_length = new_netcdf.createVariable('begin_mes_length', 'i4')
    end_mes_length = new_netcdf.createVariable('end_mes_length', 'i4')
    step_length = new_netcdf.createVariable('step_length', 'i4')
    num_repeats = new_netcdf.createVariable('num_repeats', 'i4')
    data = new_netcdf.createVariable('data', 'f8', dimensions = (n,x,y))
    noise_message_length = new_netcdf.createVariable('noise_mes_length', 'i4')
    noise_message_num = new_netcdf.createVariable('num_noise_mes', 'i4')
    noise_processors = new_netcdf.createVariable('num_noise_proc', 'i4')

    proc_num.assignValue(cdf['proc_num'].getValue())
    data_type.assignValue(cdf['data_type'].getValue())
    begin_mes_length.assignValue(0)
    end_mes_length.assignValue(1000)
    step_length.assignValue(1)
    num_repeats.assignValue(1000)
    data[:] = cur_3d
    noise_message_length.assignValue(0)
    noise_message_num.assignValue(0)
    noise_processors.assignValue(0)

    print()
    print(new_netcdf)
    print()
    print(new_netcdf['data'][:])
    

file_path = ("/home/volch/3_kurs/asvk_sc/results/" + sys.argv[1])
cdf = nc.Dataset(file_path)

print(type(cdf))
print('\n\n')
print(_("Begin length:"), cdf['begin_mes_length'][:])
print(_("Step:"), cdf['step_length'][:])
print(_("End length:"), cdf['end_mes_length'][:])


mes_length = int(input(_("Введите длину сообщения, для которой надо создать трехмерный файл: ")))
attr_length = mes_length
mes_length //= cdf['step_length'][:]

data = cdf['data'][:]

try:
    cur_3d = data[mes_length]
except:
    print(_("your length is not valid"))
else:
    cur_spektrum = spektrum(cur_3d,1,2,20,120)
    spektrum_ncdf(cur_3d, int(input(_("Введите длину окна для подсчета дискретного спектра: "))), mes_length)
    flag = input(_("Do you want to make a new netcdf cut fuli with info about latencies for given lewngth? (y/n)"))
    if flag not in ("n", "no"):
        real_3d = [[[0 for j_1 in range(len(cur_3d[0]))] for j_2 in range(len(cur_3d))] for j_3 in range(len(cur_3d[0][0]))]
        for i_1 in range(len(cur_3d)):
            for i_2 in range(len(cur_3d[0])):
                for i_3 in range(len(cur_3d[0][0])):
                    real_3d[i_3][i_1][i_2] = cur_3d[i_1][i_2][i_3]
        make_netcdf(cdf, real_3d, attr_length)
