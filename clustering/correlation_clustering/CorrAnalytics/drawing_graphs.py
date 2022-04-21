import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualise_anomalies(i, j, corr, text, threshold, meddata):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    anomaly_plot = corr.loc[i, j]
    anomaly_detectors = anomaly_plot[anomaly_plot >= threshold].sort_values()
    ax.set_title(text + ': ' +str(i) + ' -> ' + str(j))
    ax = meddata.loc[i, j].plot()
  #  print (anomaly_detectors.index.labels[1])
    for x in anomaly_detectors.index.labels[1]:
        ax.axvline(x, color='red')
    ax.set_xticks(anomaly_detectors.index.labels[1])
    ax.tick_params(axis='x', color='red', labelcolor='red', labelsize=12)
    plt.show()

def corr_analytics_non_abs(corr):
    result = pd.DataFrame(index=corr.index)
    result['0.0 < x < 0.15'] = ((corr >= 0.0) & (corr < 0.15)).sum(axis = 1)
    result['0.15 < x < 0.25'] = ((corr >= 0.15) & (corr < 0.25)).sum(axis = 1)
    result['0.25 < x < 0.35'] = ((corr >= 0.25) & (corr < 0.35)).sum(axis = 1)
    result['0.35 < x < 0.45'] = ((corr >= 0.35) & (corr < 0.45)).sum(axis = 1)
    result['0.45 < x < 0.55'] = ((corr >= 0.45) & (corr < 0.55)).sum(axis = 1)
    result['0.55 < x < 0.65'] = ((corr >= 0.55) & (corr < 0.65)).sum(axis = 1)
    result['0.65 < x < 0.75'] = ((corr >= 0.65) & (corr < 0.75)).sum(axis = 1)
    result['0.75 < x < 0.85'] = ((corr >= 0.75) & (corr < 0.85)).sum(axis = 1)
    result['0.85 < x < 1.0'] = ((corr >= 0.85) & (corr <= 1.0)).sum(axis = 1)
    return result

def analytics_diag(data, n, name):
    draw = data.sum(axis = 0)
    fig = plt.figure()
    #fig.suptitle('Распределение корреляций ' + name, fontsize=16)
    ax1 = fig.add_subplot(121)
    ax1 = draw.plot.bar(color='blue')
    ax1.set_ylabel('Количество')
    ax1.set_xlabel('Распределение')
    ax2 = fig.add_subplot(122)
    ax2 = draw[n:].plot.bar(color='blue')
    ax2.set_ylabel('Количество')
    ax2.set_xlabel('Распределение')
    fig.tight_layout(w_pad=1, h_pad=0.3)
    plt.show()

def catch_anomalies(corr, meddata, i, j, val,  title):
  #  line1 = np.full(100, -1*val)
    line2 = np.full(100, val)
    plotting_simple = corr.loc[i, j]
    corr_simple_greater = plotting_simple[plotting_simple > val]
    corr_simple_greater = corr_simple_greater.reset_index().values
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(211)
    ax1 = meddata.loc[i, j].plot(title='График задержки: ' +str(i) + '->' + str(j))
    ax2 = fig.add_subplot(212)
    str1 = ' (' + str(val) + ')'
    ax2 = corr.loc[i, j].plot(title=title+str1)
   # ax2.plot(line1, color='green')
    ax2.plot(line2, color='green')
    for i in range(0, len(corr_simple_greater[:, 1])):
        ax2.scatter(corr_simple_greater[:, 1][i], corr_simple_greater[:, 2][i], color='red', marker='x')
    fig.subplots_adjust(bottom=-0.4)
    plt.tight_layout()
    plt.show()

def cluster_draw(clusterdata, meddata, features, centroids, clust_num):
    cl_info = clusterdata[clusterdata["CLUSTER"] == clust_num]
    cl_info = cl_info.dropna()
    cl_index = cl_info.index
    smpl = cl_info.sample(5).index.labels
    ft = features[clust_num]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Кластер: ' + str(clust_num))
    for i in range(0, 5):
        ax.plot(meddata.loc[int(smpl[0][i]), int(smpl[1][i])], alpha=0.15)
    centroid = centroids.loc[clust_num]
    ax.plot(centroid, color='red', label='centroid')
    for x in ft:
        ax.axvline(x, color='lightblue')
    ax.set_xticks(ft)
    ax.tick_params(axis='x', color='lightblue', labelcolor='lightblue', labelsize=12)
    ax.legend(loc=1)
    plt.show()
