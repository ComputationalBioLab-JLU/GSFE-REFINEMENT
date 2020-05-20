"""
@author: cxy
"""

import numpy as np
import math
import sys
import time
import Bio
from Bio import PDB

t_dic={'ALA':'A','VAL':'V','LEU':'L','ILE':'I','PHE':'F','TRP':'W','MET':'M','PRO':'P',\
       'GLY':'G','SER':'S','THR':'T','CYS':'C','TYR':'Y','ASN':'N','GLN':'Q','HIS':'H',\
       'LYS':'K','ARG':'R','ASP':'D','GLU':'E'}

def rotation(r,v,theta):
    t1=r*np.cos(theta)
    t2=np.cross(v,r)
    t2=t2*np.sin(theta)
    vr=np.dot(v,r)
    t3=vr*v*(1-np.cos(theta))
    r=t1+t2+t3
    return r

def calha1(a,b,c):
    "calculate gly H coord"
    ab=b-a
    cb=b-c
    bc=c-b
    cbmo=np.linalg.norm(cb)
    d=cb*1.0814/cbmo
    bcmo=np.linalg.norm(cb)
    bc/=bcmo
    fabc=np.cross(ab,cb)
    fmo=np.linalg.norm(fabc)
    fabc/=fmo
    d=rotation(d,fabc,math.pi*108.0300/180.0)
    d=rotation(d,bc,math.pi*117.8600/180.0)
    d+=c
    return d

def get_id_chain_name(pdb_list_file):
    pdbid=[]
    pdbchain=[]
    with open(pdb_list_file, 'r') as f:
        lines = f.readlines()
    if not lines:
        print("read %s fail!" % pdb_list_file)
        sys.exit()
    for line in lines:
        line = line.strip('n')
        pdb = line.split()[0]
        pdbid.append(pdb[:4])
        pdbchain.append(pdb[4:])        
    return pdbid, pdbchain

def get_aa_list(res_list):
    aa_list = [a for a in res_list if PDB.is_aa(a)]
    return aa_list

def check_aa_id(aa_list):
    error = 0
    t=aa_list[0].get_id()[1]
    aa_list_full=[]
    for a in aa_list:
        while 1:
            if a.get_id()[1]<t:
                error=1
                break
            elif a.get_id()[1]==t:
                aa_list_full.append(a)
                t+=1
                break
            else:
                aa_list_full.append(None)
                t+=1
    if error==1:                 
        return 0
    return aa_list_full

def cal_depth(s, aa_list_full):
    depth=PDB.ResidueDepth(s)   #氨基酸到蛋白质表面距离
    dep_dict=depth.property_dict
    dps=[]
    for a in aa_list_full:
        try:
            aa_id=(a.get_parent().get_id(),a.get_id())
            if dep_dict.get(aa_id):
                dps.append(dep_dict[aa_id])
            else:
                dps.append([None,None])
        except:
            dps.append([None,None])
    dps=np.array(dps)
    return dps

def cal_hseab(s, aa_list_full):
    try:
        HSEA=PDB.HSExposureCA(s)
        HSEB=PDB.HSExposureCB(s)
    except:
        return 0,0
    HSEA_dict=HSEA.property_dict
    HSEB_dict=HSEB.property_dict
    hse_a=[]
    hse_b=[]
    for a in aa_list_full:
        try:
            aa_id=(a.get_parent().get_id(),a.get_id())
            if HSEA_dict.get(aa_id):
                hse_a.append(HSEA_dict[aa_id])
            else:
                hse_a.append([None,None,None])
        except:
            hse_a.append([None,None,None])
    hse_a=np.array(hse_a)
    for a in aa_list_full:
        try:
            aa_id=(a.get_parent().get_id(),a.get_id())
            if HSEB_dict.get(aa_id):
                hse_b.append(HSEB_dict[aa_id])
            else:
                hse_b.append([None,None,None])
        except:
            hse_b.append([None,None,None])

    hse_b=np.array(hse_b)
    return hse_a, hse_b

def get_seq(aa_list_full):
    seq_list = ''
    for a in aa_list_full:
        try:
            t=a.get_resname()
            if t in t_dic:
                seq_list+=t_dic[t]
            else:
                seq_list+='X'
        except:
            seq_list+='X'
    return seq_list

def get_atom_list(aa_list_full, atom):
    atom_list=[]
    for a in aa_list_full:
        try:
            t=a[atom]
            atom_list.append(t)
        except:
            t=None
            atom_list.append(t)
    return atom_list
        
def cal_dist(ca_list):
    ca_num=len(ca_list)
    ca_dist=[]             #CA距离
    for j in range(len(ca_list)):
        for k in range(len(ca_list)):
            if ca_list[j]!=None and ca_list[k]!=None:
                ca_dist.append(ca_list[j]-ca_list[k])
            else:
                ca_dist.append(None)    
    ca_dist=np.array(ca_dist)
    ca_dist=ca_dist.reshape(ca_num,ca_num)
    return ca_dist

def get_mask(ca_list):
    mask=[]    #是否有CA
    for j in range(len(ca_list)):
        if ca_list[j]!=None:
            mask.append(1)
        else:
            mask.append(0)
    return mask

def time_cost(start):
    time_cost = time.time() - start
    print(time_cost)

def get_cb_vector(ca_list, cb_list, n_list, c_list, j):
    if cb_list[j] != None:
        cb = cb_list[j].get_vector()
    elif cb_list[j] == None and c_list[j]!=None and n_list[j]!=None and ca_list[j]!=None:
        ca_v=ca_list[j].get_vector().get_array()
        c_v=c_list[j].get_vector().get_array()
        n_v=n_list[j].get_vector().get_array()
        cb=calha1(n_v,c_v,ca_v)
        cb=PDB.vectors.Vector(cb)
    else:
        cb = cb_list[j]
    return cb

def get_angle5_ceshi(aa_num16, ca_list, cb_list, n_list, c_list, j):
    angle_t = []
    cb1 = get_cb_vector(ca_list, cb_list, n_list, c_list, j)                
    for k in aa_num16:
        try:
            ca1 = ca_list[j].get_vector()
            ca2 = ca_list[k].get_vector()            
            t1 = PDB.vectors.calc_angle(cb1,ca1,ca2)
        except:
            t1 = None            
        try:
            ca1 = ca_list[j].get_vector()
            ca2 = ca_list[k].get_vector()
            cb2 = get_cb_vector(ca_list, cb_list, n_list, c_list, k)
            t11 = PDB.vectors.calc_angle(cb2,ca2,ca1)
        except:
            t11 = None
            
        try:
            ca1 = ca_list[j].get_vector()
            ca2 = ca_list[k].get_vector()
            n_1 = n_list[j].get_vector()
            t2 = PDB.vectors.calc_angle(n_1,ca1,ca2)
        except:
            t2 = None
        try:
            ca1 = ca_list[j].get_vector()
            ca2 = ca_list[k].get_vector()
            n_2 = n_list[k].get_vector()
            t22 = PDB.vectors.calc_angle(n_2,ca2,ca1)
        except:
            t22 = None                
        try:    
            ca1 = ca_list[j].get_vector()
            ca2 = ca_list[k].get_vector()
            c_1 = c_list[j].get_vector()
            t3 = PDB.vectors.calc_angle(c_1,ca1,ca2)
        except:
            t3 = None
        try:    
            ca1 = ca_list[j].get_vector()
            ca2 = ca_list[k].get_vector()
            c_2 = c_list[k].get_vector()
            t33 = PDB.vectors.calc_angle(c_2,ca2,ca1)
        except:
            t33 = None            
            
        angle_t = t1,t2,t3,t11,t22,t33        
        yield angle_t
