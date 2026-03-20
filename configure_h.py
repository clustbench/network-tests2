tmp = 1

with open("clustbench_config.h", "r") as f:
    tmp = f.read()

    tmp_tmp  = tmp.split(" \" \n#endif")
    print(tmp_tmp[1])

    tmp = tmp_tmp[0] + "\n#endif" + tmp_tmp[1]

    f.close()

with open("clustbench_config.h", "w") as f:
    print(tmp)
    f.write(tmp)
    f.close()
