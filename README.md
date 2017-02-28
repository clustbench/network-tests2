network-tests2
==============

Benchmarks and analysis of interconnection in HPC cluster


Deploying on Lomonosov 2
------------------------

`deploy_on_lomonosov2.mk` is a Makefile which can download and build everything
needed to run *network-tests2* on Lomonosov 2 cluster. Here is how to do it:

```shell
# use scripts available on lomonosov 2 to initialize environment variables
# these 2 commands must be run every time you start new ssh session to lom2
# you can add them to your .bashrc for convenience
$ module add openmpi/1.8.4-gcc
$ module add slurm

$ mkdir network_tests2_stuff  # create dir where everything will be built
$ cd network_tests2_stuff

$ wget --output-document Makefile https://raw.githubusercontent.com/clustbench/network-tests2/master/deploy_on_lomonosov2.mk
```

Makefile is ready to be used now. It will download all dependencies, build them
and install them to `~/_scratch/network_tests_prefix`. Lomonosov 2 is
configured in such way, that all files that are used when running your program,
must be somewhere in `~/_scratch`. It means that `~/_scratch` directory is
shared across all nodes you use, but other directories in your home folder are
not. This makefile will create directories `bin`, `include` and others in
`~/_scratch/network_tests_prefix` and set environment variables like `PATH` and
`LD_LIBRARY_PATH` accordingly, so that everything works as it should.

```shell
# download, build and install everything to _scratch/network_tests_prefix
$ make -j9 all


# run network-tests2 on 10 nodes
# you can omit NUM_NODES and RUN_DIR options
# they will default to something, see makefile to see default values
# RUN_DIR must be inside ~/_scratch
# WARNING: this makefile will delete RUN_DIR and create an empty dir in its place
$ make run NUM_NODES=10 RUN_DIR=${HOME}/_scratch/run_dir
... it will print something
Submitted batch job 123456

# you can now view info about your job
$ scontrol show job 123456
# check if it's finished
$ scontrol show job 123456 | grep -i jobstate
   JobState=PENDING Reason=JobHeldAdmin Dependency=(null)

# When execution is finished, you will be able to find output files in RUN_DIR
# the program's stdout will be in slurm-123456.output



# you can clean (delete) everything including the prefix like this
$ make clean
```

You can provide `NETWORK_TESTS2_GIT_REPO` and `NETWORK_TESTS2_BRANCH`
environment variables to use another
network_tests2 repo and/or branch. For example:

```shell
$ export NETWORK_TESTS2_GIT_REPO="https://github.com/philip-bl/network-tests2.git"
$ export NETWORK_TESTS2_BRANCH="phil_master"

# and then run all the make commands you need
```


Modify the variable `PREFIX` there to install everything to other directory.
