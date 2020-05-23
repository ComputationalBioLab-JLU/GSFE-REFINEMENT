import os
import os
import time
from tqdm import tqdm

def read_it(path_name):
    name = os.listdir(path_name)
    native_list = []
    for i in name:
        #print(i)
        native_list.append(i[:5])
    #print(len(name))
    new_list = list(set(native_list))
    return new_list

name = "native_start"
a = read_it(name)

lr = 0.0005
EN = 1
SM = 1.2
pdb_path = "native_start"
for name in tqdm(a[:3]):
    native_name = name+"_native.pdb"
    start = name+"_model.pdb"
    print("refine %s..."% native_name)
    try:
        a = os.system("python auto_opi_mutation_model770_sincos.py --LR %s --ENRAOPY_W %s --L1_smooth_parameter %s --PATH %s --native_name %s --decoy_name %s --OPI ADAM" % (lr, EN, SM, pdb_path, native_name, start))
    except:
        continue
