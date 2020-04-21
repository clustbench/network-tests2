import os
import argparse 
from links import LinkList
import netCDF4 as nc

parser = argparse.ArgumentParser()

parser.add_argument('--input_folder', type=str, default='',help='Folder with')
parser.add_argument('--eq_classes', type=str, default='classes.csv',help='CSV file with information about links equality classes')
parser.add_argument('--output_prefix', type=str, default='estimate_network_',help='Prefix of output NetCDF4 files with results of prediction')

args = parser.parse_args()

beg_mes_len = 0
end_mes_len = 0
step_length = 0
repeats = 0

data_map = {}
hosts_map = {}

for filename in os.listdir(args.input_folder):
    if filename.endswith('.nc'):
        cls = int(filename.split('_')[0])
        data = nc.Dataset(args.input_folder + '/' + filename)
        delays = data.variables['data']
        beg_mes_len = data.variables['begin_mes_length'].getValue()
        end_mes_len = data.variables['end_mes_length'].getValue()
        step_length = data.variables['step_length'].getValue()
        repeats = data.variables['num_repeates'].getValue()
        data_map[cls] = delays 

    if filename.endswith('.txt'):
        k = 0
        with open(args.input_folder + '/' + filename, 'r') as f:
            for line in f:
                hosts_map[line.strip()] = k
                k += 1

links = LinkList(args.eq_classes, data_map, hosts_map)
links.populate_delays(repeats, beg_mes_len, end_mes_len, step_length)
links.save_to_netcdf(args.output_prefix, beg_mes_len, end_mes_len, step_length)


