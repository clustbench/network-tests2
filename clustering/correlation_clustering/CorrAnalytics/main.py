import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import sys
import drawing_graphs

global mode
global corr
global meddata
global name
global clusters
global centroids
global features

def load_from_file():
    global meddata
    global corr
    global name
    global clusters
    global centroids
    global mode
    global features

    if len(sys.argv) != 4:
        print ("Wrong arguments. Usage: <network_median.nc> <result.h5> <mode>")
        exit(0)
    medians = sys.argv[1]
    ds = xr.open_dataset(medians)
    df = ds.to_dataframe()
    data = df['data']
    meddata = data.unstack(0)
    results = h5py.File(sys.argv[2], 'r')
    if sys.argv[3] == "corr":
        mode = "corr"
        name = results.attrs['TYPE']
        corr_data = results['CORR']
        corr_data = np.copy(corr_data)
        corr_data = xr.DataArray(corr_data, dims=['x', 'y', 'n'])
        corr = corr_data.to_dataframe(name='CORRS')
        corr = corr.unstack(2)
    if sys.argv[3] == "clust":
        mode = "clust"
        centroids = pd.DataFrame(columns=meddata.columns)
        num_proc = results.attrs['NUM_PROC'][0]
        k = results.attrs['K'][0]
        clusters = pd.DataFrame(index=meddata.index, columns=["CLUSTER"])
        features = []
        for i in range(0, k):
            clust_string = "CLUSTER_" + str(i)
            elems = np.array(results[clust_string]["CLUSTER_ELEMENTS"])
            centroids.loc[i] = np.array(results[clust_string]["CENTROID"])
            features.append(np.array(results[clust_string]["FEATURES"]))
            for j in range(0, len(elems)):
                clusters.loc[elems[j][0], elems[j][1]]["CLUSTER"] = int(i)
            

def draw_features():
    i = input("Sender: ")
    j = input("Reciever: ")
    thres = input("Threshold: ")
    drawing_graphs.visualise_anomalies(int(i), int(j), corr, str(name), float(thres), meddata)

def draw_corr_distribution():
    corr_analytics = drawing_graphs.corr_analytics_non_abs(corr)
    drawing_graphs.analytics_diag(corr_analytics, 2, str(name))

def draw_featlength_distributuion():
    threshold = input("Threshold: ")
    s = corr[corr >= float(threshold)]
    s = s.count()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = s.plot.bar(color='blue')
    a = ax.get_xticks()
    a = a[a % 10 == 0]
    ax.set_xticks(a)
    ax.set_xticklabels([str(ss) for ss in a])
   # ax.set_title('Особенностей на длину сообщения')
    ax.set_xlabel('Длина сообщения 10^-2')
    ax.set_ylabel('Количество')
    plt.show()

def draw_featpair_distribution():
    corr_analytics = drawing_graphs.corr_analytics_non_abs(corr)
    s = corr_analytics.iloc[:, 2:].sum(axis=1)
    q = s.value_counts()
   # fig =  plt.figure()
    rects = plt.bar(q.index, q.iloc[:])
    #plt.title('Соотношение количества особенностей', fontsize=20)
    plt.xlabel('Кол-во особенностей', fontsize=16)
    plt.ylabel('Кол-во пар', fontsize=16)
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., height,
                 '%d' % int(height),
                 ha='center', va='bottom')
    plt.show()

def draw_thres_analysis():
    i = input('Sender: ')
    j = input('Reciever: ')
    thres = input("Threshold: ")
    title = input("Title: ")
    drawing_graphs.catch_anomalies(corr, meddata, int(i), int(j), float(thres), title)

def draw_cluster():
    i = input('Number of cluster: ')
    drawing_graphs.cluster_draw(clusters, meddata, features, centroids, int(i))

load_from_file()

executables_corr = {
    'exit': exit,
    'drawfeat': draw_features,
    'corrdistr': draw_corr_distribution,
    'lendistr': draw_featlength_distributuion,
    'pairdistr': draw_featpair_distribution,
    'analysethres': draw_thres_analysis,
}

executables_clust = {
    'exit': exit,
    'cluster': draw_cluster
}

command = ""
while (1):
    command = input("Command: ")
    if (mode == "corr"):
        executables_corr[command]()
    if (mode == "clust"):
        executables_clust[command]()