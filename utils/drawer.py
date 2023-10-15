import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import sys
import netCDF4 as nc
import pandas as pd
import gettext


def autolabel(rects, labels=None, height_factor=1.01):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if labels is not None:
            try:
                label = labels[i]
            except (TypeError, KeyError):
                label = ' '
        else:
            label = '%d' % int(height)
        ax.text(rect.get_x() + rect.get_width()/2., height_factor*height,
                '{}'.format(label),
                ha='center', va='bottom')

mode = int(input("1: compare, 2: algo_stats, 3: check data"))
if mode == 1:
    with open("log_algo_single.txt", "r") as f:
        json_in = json.loads(f.readline())

    fin = len(json_in['iters'])
    for i in range(0, fin, 2):
        if json_in['iters'][i] > json_in['iters'][i+1]:
            for k in json_in:
                json_in[k][i], json_in[k][i+1] = json_in[k][i+1], json_in[k][i]

    tested_configs = set(json_in['config'])
    s = ['green','red','blue', 'magenta']
    letters = ['a', 'b', 'c']
    acc_coeffs = []
    for i in range(len(json_in['acc_params'])):
        if len(json_in['acc_params'][i]) != 1:
            acc_coeffs.append(json_in['acc_params'][i][1])
        else:
            acc_coeffs.append(json_in['acc_params'][i][0])

    colors = [s[i//(len(json_in['iters'])//len(tested_configs))] for i in range(len(json_in['iters']))]
    for param in ('iters', 'acc_min', 'acc_dev', 'acc_avg', 'acc_med'):
        y = json_in[param]
        x = [i for i in range(len(y))]
        colors = [s[i//(len(y)//len(tested_configs))] for i in range(len(y))]

        plt.rc('font', size=30)
        ##plt.rc('xtick', labelsize=24)
        plt.figure(figsize = (170,120))
        plt.title(param)
        plt.xlabel("алгоритм и набор параметров")
        plt.grid(True)
        letters = ['a', 'b', 'c']
        plt.xticks(range(0, 12), [f"{letters[(i//2)%3]}\n{i%2+1}" for i in range(12)], rotation = 'horizontal')
        to_draw = sorted(y)
        new_draw = [to_draw[0]]
        prev = to_draw[0]
        for i in range(1, len(to_draw)-1):
            if to_draw[i]-prev > 0.035*to_draw[-1]:
                new_draw.append(to_draw[i])
                prev = to_draw[i]
        new_draw.append(to_draw[-1])
        plt.yticks(new_draw, rotation = 'horizontal')
        plt.bar(x, y, color = colors,)
        ax = plt.gca()
        autolabel(ax.patches, acc_coeffs, height_factor=1.01)
        plt.grid(axis = 'x')
        plt.show()

if mode == 2:
    with open("log_algo_multi.txt", "r") as f:
        json_launches = json.loads(f.readline())

    for i in range(len(json_launches['launches'])):
        launch = json_launches['launches'][i]
        x = sorted(list(json_launches['iters'][i].keys()), key = lambda tmp_cur: int(tmp_cur))
        print(x)
        y = [json_launches['iters'][i][k] for k in x]
        plt.rc('font', size=30)
        plt.xticks(range(0, len(x)), x)
        plt.yticks(None)
        plt.ylabel("Количество соответсвующих результатов")
        plt.xlabel("Количество итераций")
        plt.bar(range(0, len(x)), y)
        ax = plt.gca()
        autolabel(ax.patches, y, height_factor=1.01)
        plt.show()
        stats_accs = {'acc_min': [], 'acc_avg': [], 'acc_med': [], 'acc_dev': []}
        for k in stats_accs:
            if k == 'acc_dev':
                stats_accs[k] = [json_launches[k][i][n_iters]/100 for n_iters in x]
            else:
                stats_accs[k] = [json_launches[k][i][n_iters] for n_iters in x]
        df = pd.DataFrame(stats_accs)
        plt.xticks(range(0, len(x)), x)
        plt.ylabel("Относительные погрешности")
        plt.xlabel("Количество итераций")
        plt.plot(range(0, len(x)), df, linewidth = 3)
        plt.grid(axis = 'x')
        my_legend = ('acc_min', 'acc_avg', 'acc_med', 'acc_dev/100')
        plt.legend(my_legend, loc=1)
        plt.show()

if mode == 3:
    plt.rc('font', size=30)
    file_path = ("/home/volch/3_kurs/asvk_sc/results/" + sys.argv[1])
    cdf = nc.Dataset(file_path)

    print(_("Begin length:"), cdf['begin_mes_length'][:])
    print(_("Step:"), cdf['step_length'][:])
    print(_("End length:"), cdf['end_mes_length'][:])
    print(_("Proc amount:"), cdf['proc_num'][:])
    while (params := input("Insert message length, source number and destination number to run the algorithm on: ")) != 'quit':
        params = params.split()
        print(params)
        mes_length = int(params[0])
        mes_length //= cdf['step_length'][:]
        n_src = int(params[1])
        n_dest = int(params[2])

        data = cdf['data'][:]

        try:
            cur_1d = data[mes_length][n_src][n_dest]
        except IndexError:
            print("your params are not valid")
            continue

        target_min = min(cur_1d)
        target_avg = sum(cur_1d)/len(cur_1d)
        target_dev = sum([i**2 for i in cur_1d])/len(cur_1d)-target_avg**2
        target_med = sorted(cur_1d)[len(cur_1d)//2]
        mins = []
        avgs = []
        devs = []
        meds = []
        frequency = int(input("How often shall we count stats:"))
        iters = [i for i in range(len(cur_1d)//frequency, len(cur_1d), len(cur_1d)//frequency)]
        for i in range(frequency-1):
            res_vector = cur_1d[:iters[i]-1]
            res_min = min(res_vector)
            res_avg = sum(res_vector)/len(res_vector)
            res_dev = sum([i**2 for i in res_vector])/len(res_vector)-res_avg**2
            res_med = sorted(res_vector)[len(res_vector)//2]
            mins.append(abs(1-res_min/target_min))
            avgs.append(abs(1-res_avg/target_avg))
            devs.append(abs(1-res_dev/target_dev))
            meds.append(abs(1-res_med/target_med))
        titles = ['min', 'avg', 'dev', 'med']
        j = 0
        for params in (mins, avgs, devs, meds):
            plt.xticks(range(0, len(iters)), iters)
            to_draw = sorted(params)
            new_draw = [to_draw[0]]
            prev = to_draw[0]
            for i in range(1, len(to_draw)-1):
                if to_draw[i]-prev > 0.035*to_draw[-1]:
                    new_draw.append(to_draw[i])
                    prev = to_draw[i]
            new_draw.append(to_draw[-1])
            plt.yticks(new_draw)
            plt.title(titles[j])
            plt.xlabel("количество итераций")
            plt.ylabel("относительная погршеность")
            plt.bar(range(0, len(iters)), params)
            plt.grid(axis = 'y')
            plt.show()
            j += 1
