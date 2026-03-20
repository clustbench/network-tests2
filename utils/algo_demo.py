import netCDF4 as nc
import sys
from scipy.fft import rfft, rfftfreq
import numpy as np
from matplotlib import pyplot as plt
import random
import numpy.linalg
import json
import gettext


global_window_params = []
global_acc_params = []


# 1
def algo_stat_diff_eval(cur_1d, flag_log):
    global global_acc_params, global_window_params
    max_iter = global_window_params[0]
    frequency = global_window_params[1]
    window_amount = global_window_params[2]
    window_sum_length = global_window_params[3]
    
    if global_acc_params[0] == -1:
        min_stat = False
    else:
        min_stat = True
        k_min = global_acc_params[0]
        
    if global_acc_params[1] == -1:
        dev_stat = False
    else:
        dev_stat = True
        k_dev = global_acc_params[1]
        
    if global_acc_params[2] == -1:
        med_stat = False
    else:
        med_stat = True
        k_med = global_acc_params[2]
        
    if global_acc_params[3] == -1:
        avg_stat = False
    else:
        avg_stat = True
        k_avg = global_acc_params[3]

    iter_step = max_iter//frequency
    cur_iter = iter_step

    while cur_iter <= max_iter:
        window_length = window_sum_length*cur_iter//window_amount + 1
        windows = [random.randint(0, cur_iter-window_length) for _ in range(window_amount)]
        first_w = cur_1d[windows[0]:windows[0]+window_length]
        min_max_stats = {}
        if min_stat:
            param = min(first_w)
            min_max_stats['min'] = {'min': param, 'max': param}
        if avg_stat:
            param = sum(first_w)/window_length
            min_max_stats['avg'] = {'min': param, 'max': param}
        if dev_stat:
            param = sum([i**2 for i in first_w])/window_length-(sum(first_w)/window_length)**2
            min_max_stats['dev'] = {'min': param, 'max': param}
        if med_stat:
            param = sorted(first_w)[window_length//2]
            min_max_stats['med'] = {'min': param, 'max': param}

        for w in windows[1:]:
            cur_w = cur_1d[w:w+window_length]
            if min_stat:
                param = min(cur_w)
                if param < min_max_stats['min']['min']:
                    min_max_stats['min']['min'] = param
                elif param > min_max_stats['min']['max']:
                    min_max_stats['min']['max'] = param
            if avg_stat:
                param = sum(cur_w)/window_length
                if param < min_max_stats['avg']['min']:
                    min_max_stats['avg']['min'] = param
                elif param > min_max_stats['avg']['max']:
                    min_max_stats['avg']['max'] = param
            if dev_stat:
                param = sum([i**2 for i in cur_w])/window_length-(sum(cur_w)/window_length)**2
                if param < min_max_stats['dev']['min']:
                    min_max_stats['dev']['min'] = param
                elif param > min_max_stats['dev']['max']:
                    min_max_stats['dev']['max'] = param
            if med_stat:
                param = sorted(cur_w)[window_length//2]
                if param < min_max_stats['med']['min']:
                    min_max_stats['med']['min'] = param
                elif param > min_max_stats['med']['max']:
                    min_max_stats['med']['max'] = param

        exit_condition = True
        if min_stat:
            param_min = min_max_stats['min']['min']
            param_max = min_max_stats['min']['max']
            param_res = param_min/param_max
            if flag_log:
                print(_(f"min_koeff on {cur_iter} iterations is: {param_res} counted on windows with length {window_length}"))
            if param_res < k_min:
                exit_condition = False
        if dev_stat:
            param_min = min_max_stats['dev']['min']
            param_max = min_max_stats['dev']['max']
            param_res = param_min/param_max
            if flag_log:
                print(_(f"dev_koeff on {cur_iter} iterations is: {param_res} counted on windows with length {window_length}"))
            if param_res < k_dev:
                exit_condition = False
        if med_stat:
            param_min = min_max_stats['med']['min']
            param_max = min_max_stats['med']['max']
            param_res = param_min/param_max
            if flag_log:
                print(_(f"med_koeff on {cur_iter} iterations is: {param_res} counted on windows with length {window_length}"))
            if param_res < k_med:
                exit_condition = False
        if avg_stat:
            param_min = min_max_stats['avg']['min']
            param_max = min_max_stats['avg']['max']
            param_res = param_min/param_max
            if flag_log:
                print(_(f"avg_koeff on {cur_iter} iterations is: {param_res} counted on windows with length {window_length}"))
            if param_res < k_avg:
                exit_condition = False

        cur_iter += iter_step
        if exit_condition:
            break

    if flag_log:
        print(_(f'Result has been reached within {cur_iter-iter_step} iterations, max_iters was {max_iter}'))
    if flag_log:
        print(_('Reached quality measurements:'))
    if min_stat:
        param_min = min_max_stats['min']['min']
        param_max = min_max_stats['min']['max']
        if flag_log:
            print(_(f' {param_min/param_max} for min'))
    if dev_stat:
        param_min = min_max_stats['dev']['min']
        param_max = min_max_stats['dev']['max']
        if flag_log:
            print(_(f' {param_min/param_max} for dev, k_dev was {k_dev}'))
    if med_stat:
        param_min = min_max_stats['med']['min']
        param_max = min_max_stats['med']['max']
        if flag_log:
            print(_(f' {param_min/param_max} for med'))
    if avg_stat:
        param_min = min_max_stats['avg']['min']
        param_max = min_max_stats['avg']['max']
        if flag_log:
            print(_(f' {param_min/param_max} for avg'))
    return cur_iter-iter_step


# 2
def algo_spektr_diff_eval(cur_1d, flag_log):
    global global_acc_params, global_window_params
    
    max_iter = global_window_params[0]
    frequency = global_window_params[1]
    window_amount = global_window_params[2]
    window_sum_length = global_window_params[3]
    accuracy = global_acc_params[0]

    iter_step = max_iter//frequency
    cur_iter = iter_step

    while cur_iter <= max_iter:
        window_length = window_sum_length*cur_iter//window_amount + 1
        windows = [random.randint(0, cur_iter-window_length) for _ in range(window_amount)]
        first_w = cur_1d[windows[0]:windows[0]+window_length]

        spektrums = []
        norms_sum = 0
        dists_sum = 0
        for w in windows:
            cur_w = cur_1d[w:w+window_length]
            yf = np.abs(rfft(cur_w))
            norms_sum += numpy.linalg.norm(yf, ord=2)
            for tmp in spektrums:
                dists_sum += numpy.linalg.norm(tmp - yf, ord=2)
            spektrums.append(yf)
        cur_acc = 1-(dists_sum/(window_amount**2-window_amount))/(norms_sum/window_amount)
        if flag_log:
            print(_(f"Accuracy reached within {cur_iter} iterations: {cur_acc}"))
        cur_iter += iter_step
        if cur_acc > accuracy:
            break

    if flag_log:
        print(_(f'Result has been reached within {cur_iter-iter_step} iterations, max_iters was {max_iter}'))
    if flag_log:
        print(_(f'reached accuracy: {cur_acc}'))
    return cur_iter-iter_step

#3
def algo_spektr_mode_diff_eval(cur_1d, flag_log):
    global global_acc_params, global_window_params

    max_iter = global_window_params[0]
    frequency = global_window_params[1]
    window_amount = global_window_params[2]
    window_sum_length = global_window_params[3]
    accuracy = global_acc_params[0]

    iter_step = max_iter//frequency
    cur_iter = iter_step

    while cur_iter <= max_iter:
        window_length = window_sum_length*cur_iter//window_amount + 1
        windows = [random.randint(0, cur_iter-window_length) for _ in range(window_amount)]
        first_w = cur_1d[windows[0]:windows[0]+window_length]

        spektrums = []
        norms_sum = 0
        dists_sum = 0
        for w in windows:
            cur_w = cur_1d[w:w+window_length]
            yf = np.abs(rfft(cur_w))
            yf = yf[1:]
            norms_sum += numpy.linalg.norm(yf, ord=2)
            for tmp in spektrums:
                dists_sum += numpy.linalg.norm(tmp - yf, ord=2)
            spektrums.append(yf)
        cur_acc = (dists_sum/(window_amount**2-window_amount))/(norms_sum/window_amount)
        if flag_log:
            print(_(f"Accuracy reached within {cur_iter} iterations: {cur_acc}"))
        cur_iter += iter_step
        if cur_acc < accuracy:
            break

    if flag_log:
        print(_(f'Result has been reached within {cur_iter-iter_step} iterations, max_iters was {max_iter}'))
    if flag_log:
        print(_(f'reached accuracy: {cur_acc}'))
    return cur_iter-iter_step

# 4
def algo_vector_diff_eval(cur_1d):
    global global_acc_params, global_window_params

    max_iter = min(int(input(_('Insert max. allowed number of iterations delay measurement: '))), len(cur_1d))
    frequency = int(input(_('Insert frequency of stop condition checking (how many times algo will check the condition before reaching max_iter): ')))
    window_amount = int(input(_('Insert amount of windows to count the stats on: ')))
    window_sum_length = int(input(_('Insert the ratio of the total length of the windows to the current number of iterations: ')))
    accuracy = float(input(_('Insert accuracy')))

    iter_step = max_iter//frequency
    cur_iter = iter_step

    while cur_iter <= max_iter:
        window_length = window_sum_length*cur_iter//window_amount + 1
        windows = [random.randint(0, cur_iter-window_length) for _ in range(window_amount)]
        first_w = cur_1d[windows[0]:windows[0]+window_length]

        cur_windows = []
        norms_sum = 0
        dists_sum = 0
        for w in windows:
            cur_w = cur_1d[w:w+window_length]
            norms_sum += numpy.linalg.norm(cur_w, ord=2)
            for tmp in cur_windows:
                dists_sum += numpy.linalg.norm(tmp - cur_w, ord=2)
            cur_windows.append(cur_w)
        cur_acc = (dists_sum/(window_amount**2-window_amount))/(norms_sum/window_amount)
        print(_(f"Accuracy reached within {cur_iter} iterations: {cur_acc}"))
        cur_iter += iter_step
        if cur_acc < accuracy:
            break

    print(_(f'Result has been reached within {cur_iter-iter_step} iterations, max_iters was {max_iter}'))
    print(_(f'reached accuracy: {cur_acc}'))
    return cur_iter-iter_step


def make_params(algo, cur_1d):
    global global_acc_params, global_window_params
    if algo == 1:
        min_stat = (input(_('Do you want to use min stat?(y/n)')) in ('y', 'Y', 'Yes', 'yes'))
        dev_stat = (input(_('Do you want to use dev stat?(y/n)')) in ('y', 'Y', 'Yes', 'yes'))
        med_stat = (input(_('Do you want to use med stat?(y/n)')) in ('y', 'Y', 'Yes', 'yes'))
        avg_stat = (input(_('Do you want to use avg stat?(y/n)')) in ('y', 'Y', 'Yes', 'yes'))
        max_iter = min(int(input(_('Insert max. allowed number of iterations delay measurement: '))), len(cur_1d))
        frequency = int(input(_('Insert frequency of stop condition checking (how many times algo will check the condition before reaching max_iter): ')))
        window_amount = int(input(_('Insert amount of windows to count the stats on: ')))
        window_sum_length = int(input(_('Insert the ratio of the total length of the windows to the current number of iterations: ')))
        global_acc_params = []
        if min_stat:
            k_min = float(input(_('Insert coeff. for min: ')))
            global_acc_params.append(k_min)
        else:
            global_acc_params.append(-1)
        if dev_stat:
            k_dev = float(input(_('Insert coeff. for dev: ')))
            global_acc_params.append(k_dev)
        else:
            global_acc_params.append(-1)
        if med_stat:
            k_med = float(input(_('Insert coeff. for med: ')))
            global_acc_params.append(k_med)
        else:
            global_acc_params.append(-1)
        if avg_stat:
            k_avg = float(input(_('Insert coeff. for avg: ')))
            global_acc_params.append(k_avg)
        else:
            global_acc_params.append(-1)

        global_window_params = [max_iter, frequency, window_amount, window_sum_length]

        return
    else:
        max_iter = min(int(input(_('Insert max. allowed number of iterations delay measurement: '))), len(cur_1d))
        frequency = int(input(_('Insert frequency of stop condition checking (how many times algo will check the condition before reaching max_iter): ')))
        window_amount = int(input(_('Insert amount of windows to count the stats on: ')))
        window_sum_length = int(input(_('Insert the ratio of the total length of the windows to the current number of iterations: ')))
        accuracy = float(input(_('Insert accuracy')))

        global_acc_params = [accuracy]
        global_window_params = [max_iter, frequency, window_amount, window_sum_length]


# launch
file_path = (sys.argv[1])
cdf = nc.Dataset(file_path)

print("KEYS", cdf)
input()

print(_("Begin length:"), cdf['begin_mes_length'][:])
print(_("Step:"), cdf['step_length'][:])
print(_("End length:"), cdf['end_mes_length'][:])
print(_("Proc amount:"), cdf['proc_num'][:])

available_algorithms = {1: [_('algorithm based on the estimation of the spread of statistical values'), algo_stat_diff_eval],
                        2: [_('algorithm based on the estimation of the spektrum distancies'), algo_spektr_diff_eval],
                        3: [_('algorithm based on the estimation of the spektrum distancies with powerfull harmonics separated.'), algo_spektr_mode_diff_eval],
                        4: [_('algorithm based on the estimation of distancies between vectors itself'), algo_vector_diff_eval]}


while (params := input(_("Insert message length, source number and destination number to run the algorithm on: "))) != 'quit':
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
        print(_("your params are not valid"))
        continue

    target_min = min(cur_1d)
    target_avg = sum(cur_1d)/len(cur_1d)
    target_dev = sum([i**2 for i in cur_1d])/len(cur_1d)-target_avg**2
    target_med = sorted(cur_1d)[len(cur_1d)//2]
    print(_('Choose algorithm you want to test (insert number) and how many times to run: '))
    for algo in available_algorithms:
        print(f"{algo}) {available_algorithms[algo][0]}")
    launch = input().split()
    algo = int(launch[0])
    make_params(algo, cur_1d)
    launches = (1 if len(launch) == 1 else int(launch[1]))
    print(launches)
    flag = True
    for _ in range(launches):
        res = available_algorithms[algo][1](cur_1d, launches==1)
        res_vector = cur_1d[:res]
        res_min = min(res_vector)
        res_avg = sum(res_vector)/len(res_vector)
        res_dev = sum([i**2 for i in res_vector])/len(res_vector)-res_avg**2
        res_med = sorted(res_vector)[len(res_vector)//2]
        acc_min = abs(1-res_min/target_min)
        acc_avg = abs(1-res_avg/target_avg)
        acc_dev = abs(1-res_dev/target_dev)
        acc_med = abs(1-res_med/target_med)
        if (launches == 1):
            print(_("RESULT RELATIVE ACCURACIES:"))
            print(_(f"MIN: {acc_min}"))
            print(_(f"AVG: {acc_avg}"))
            print(_(f"DEV: {acc_dev}"))
            print(_(f"MED: {acc_med}"))
            file_name = 'log_algo_single.txt'
            with open(file_name, "r") as f:
                json_in = json.loads(f.readline())
                json_in['config'].append(sys.argv[1])
                json_in['coords'].append(params)
                json_in['used_algo'].append(algo)
                json_in['window_params'].append(global_window_params)
                json_in['acc_params'].append(global_acc_params)
                json_in['iters'].append(res)
                json_in['acc_min'].append(acc_min)
                json_in['acc_avg'].append(acc_avg)
                json_in['acc_dev'].append(acc_dev)
                json_in['acc_med'].append(acc_med)
                json_out = json.dumps(json_in)
            with open(file_name, "w") as f:
                f.write(json_out)
        else:
            file_name = 'log_algo_multi.txt'
            with open(file_name, "r") as f:
                json_launches = json.loads(f.readline())
                cur_len = len(json_launches['launches'])
                if flag:
                    json_launches['launches'].append(launches)
                    cur_len += 1
                    json_launches['config'].append(sys.argv[1])
                    json_launches['coords'].append(params)
                    json_launches['used_algo'].append(algo)
                    json_launches['window_params'].append(global_window_params)
                    json_launches['acc_params'].append(global_acc_params)
                    frequency = global_window_params[1]
                    tmp_iters = [i for i in range(len(cur_1d)//frequency, len(cur_1d), len(cur_1d)//frequency)]
                    json_launches['acc_min'].append({})
                    json_launches['acc_avg'].append({})
                    json_launches['acc_dev'].append({})
                    json_launches['acc_med'].append({})
                    for i in range(frequency-1):
                        res_vector = cur_1d[:tmp_iters[i]-1]
                        res_min = min(res_vector)
                        res_avg = sum(res_vector)/len(res_vector)
                        res_dev = sum([i**2 for i in res_vector])/len(res_vector)-res_avg**2
                        res_med = sorted(res_vector)[len(res_vector)//2]
                        k = str(tmp_iters[i])
                        json_launches['acc_min'][cur_len-1][k] = (abs(1-res_min/target_min))
                        json_launches['acc_avg'][cur_len-1][k] = (abs(1-res_avg/target_avg))
                        json_launches['acc_dev'][cur_len-1][k] = (abs(1-res_dev/target_dev))
                        json_launches['acc_med'][cur_len-1][k] = (abs(1-res_med/target_med))
                    json_launches['iters'].append({})
                    flag = False
                json_launches['iters'][cur_len-1].setdefault(str(res), 0)
                json_launches['iters'][cur_len-1][str(res)] += 1
                json_launches_out = json.dumps(json_launches)
            with open(file_name, "w") as f:
                f.write(json_launches_out)
                  
    
