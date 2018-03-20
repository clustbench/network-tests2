HPC-latencies-prediction
========================

 A tool for predicting network latencies in HPC clusters using informationg about its topology and limited amount of ping data.

How to Use
----------

0. Use Python 3. `pip install openpyxl networkx pandas libcrap numpy path.py netCDF4 click`
1. Put wire journal into this directory, run <./lom2.ipynb>. It will parse the topology of Lomonosov 2 and write the graph in pickle format to the disk.
2. Run <./lom2-agnostic-code.ipynb>. It doesn't care about Lomonosov 2, it needs pickle files of the graph, which we have generated in the previous step. This step divides all node pairs into classes and again writes them to the disk as pickle files.
3. Run `python3 validate.py --verbose --tests-results=/path/to/datasets`. The directory passed as the argument must have multiple nt2 datasets in it. The directory structure should be like this:

```
datasets/
├── 2016-11-04-lom2_100_nodes
│   ├── network_average.nc
│   ├── network_deviation.nc
│   ├── network_hosts.txt
│   ├── network_median.nc
│   ├── network_min.nc
│   └── slurm-103615.out
├── 2017-02-10__110_nodes
│   ├── network_average.nc
│   ├── network_deviation.nc
│   ├── network_hosts.txt
│   ├── network_median.nc
│   ├── network_min.nc
│   └── slurm-181655.out
├── 2017-02-12__118_nodes
│   ├── network_average.nc
│   ├── network_deviation.nc
│   ├── network_hosts.txt
│   ├── network_median.nc
│   ├── network_min.nc
│   └── slurm-184120.out
├── 2017-04-29__25_nodes_01
│   ├── network_average.nc
│   ├── network_deviation.nc
│   ├── network_hosts.txt
│   ├── network_median.nc
│   ├── network_min.nc
│   └── slurm-276944.out
└── 2017-04-29__75_nodes_03
    ├── network_average.nc
    ├── network_deviation.nc
    ├── network_hosts.txt
    ├── network_median.nc
    ├── network_min.nc
    └── slurm-276946.out
```

If you want to actually use predictions and not just validate how good they are, then look at code in the end of <./predict.py>.