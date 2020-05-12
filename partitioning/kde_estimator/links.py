import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import netCDF4 as nc
import random

class LinkList:

    class Link:
        def __init__(self, send, recv, cls, hops = 0, delays = None):
            self.send = send
            self.recv = recv
            self.cls = cls 
            self.delays = delays
            self.hops = hops


    def __init__(self, csv_file, data_map, hosts_map, dropout=False):
       
        flatten_map = {}

        self.links = []

        self.hosts = []

        self.all_delays_average = []

        self.estimators = {}

        random.seed(42)
        with open(csv_file, 'r') as csv_f:
            readCSV = csv.reader(csv_f, delimiter=',')
            skip = True
            for row in readCSV:
                if skip:
                    skip = False
                    continue
                
                dropconn = False

                if dropout:
                    dropconn = random.randint(0,1) == 1

                cls = int(row[0].split(':')[0])
                hops = 0
                if (len(row[0].split(':'))):
                    hops = int(row[0].split(':')[1]) 
                if row[1] in hosts_map:

                    send = hosts_map[row[1]]
                    if row[2] in hosts_map:
                        recv = hosts_map[row[2]]
                        
                        print(dropconn)
                        if (cls in data_map) and (dropconn == False):
                            delays = np.squeeze(np.array(data_map[cls][:, send, recv, :]))
                            if cls not in flatten_map:
                                flatten_map[cls] = delays
                            else:
                                flatten_map[cls] = np.concatenate((flatten_map[cls], delays), axis = 1)
                            app_link = self.Link(row[1], row[2], cls, hops, delays)
                            self.links.append(app_link)
                        else:
                            app_link = self.Link(row[1], row[2], cls, hops)
                            self.links.append(app_link)
                    else:
                        app_link = self.Link(row[1], row[2], cls, hops)
                        self.links.append(app_link)
                else:
                    app_link = self.Link(row[1], row[2], cls, hops)
                    self.links.append(app_link)
                
                if row[1] not in self.hosts:
                    self.hosts.append(row[1])

                if row[2] not in self.hosts:
                    self.hosts.append(row[2])

        for cls in flatten_map.keys():
            self.estimators[cls] = []
            for mes_len_idx in range(0, flatten_map[cls].shape[0]):
                est = KernelDensity(bandwidth = 1e-6, kernel = 'gaussian')
                data = flatten_map[cls][mes_len_idx, :].reshape(-1, 1)
                est.fit(data)
                self.estimators[cls].append((est, data))


                
    def __estimate_delay(self, cls, repeats, total_mes):
        print('have data')
        delays = np.empty(total_mes, dtype=np.float32)
        for mes_len in range(0, total_mes):
            estimator = self.estimators[cls][mes_len][0]
            samples  = self.estimators[cls][mes_len][0].sample(repeats, random_state=0)
            #delays[mes_len] = samples[np.argmax(estimator.score_samples(samples))]
            delays[mes_len] = np.mean(samples)

        return delays 

    def __estimate_delay2(self, hops, band, begin_mes_length, end_mes_length, step):
        print('No class data!')
        total_mes = (end_mes_length - begin_mes_length) // step
        delays = np.array([l * 8 for l in range(begin_mes_length, end_mes_length, step)], dtype=np.float32)
        delays /= (band * 1024 * 1024 * 1024)
       # print(delays, hops)
        return delays * hops


    def save_to_netcdf(self, prefix_name, begin_mes_length, end_mes_length, step):
        dataset = nc.Dataset(prefix_name + "avg.nc", "w")

        bml = dataset.createVariable('begin_mes_length', 'i4')
        eml = dataset.createVariable('end_mes_length', 'i4')
        sl = dataset.createVariable('step_length', 'i4')
        pn = dataset.createVariable('proc_num', 'i4')

        bml[:] = begin_mes_length
        eml[:] = end_mes_length
        sl[:] = step
        pn[:] = self.all_delays_average.shape[1]

        x = dataset.createDimension('x', self.all_delays_average.shape[1])
        y = dataset.createDimension('y', self.all_delays_average.shape[2])
        n = dataset.createDimension('n', None)
        data = dataset.createVariable('data', 'f4', ('n', 'x', 'y'))
        for i in range(0, self.all_delays_average.shape[0]):
            data[i, :, :] = self.all_delays_average[i, :, :]

        dataset.close()

        with open(prefix_name + 'hosts.txt', 'w') as f:
            for host in self.hosts:
                f.write(host + '\n')

    def populate_delays(self, iterations, begin_message_length, end_message_length, step):
        total_mes = (end_message_length - begin_message_length) // step

        self.all_delays_average = np.full((total_mes, len(self.hosts), len(self.hosts)), -1.0, dtype=np.float32)        

        k = 0
        lst = []
        for link in self.links:

            send_idx = self.hosts.index(link.send)
            recv_idx = self.hosts.index(link.recv)

            if link.delays is not None:
                lst.append((send_idx, recv_idx))
                self.all_delays_average[:, send_idx, recv_idx] = np.average(link.delays, axis = 1)
            else:
                if (link.cls in self.estimators.keys()):
                    delays = self.__estimate_delay(link.cls, 50, total_mes)
                else:
                    delays = self.__estimate_delay2(link.hops, 56, begin_message_length, end_message_length, step)
               # print(delays)
                self.all_delays_average[:, send_idx, recv_idx] = delays
                self.all_delays_average[:, recv_idx, send_idx] = delays

            k += 1
            print("Done {:d} from {:d}".format(k, len(self.links)))