"from pdb structure extract feature include: ca mask, ca distance, angle, hsea, hseb, residue depth. save .npy file "

#from Bio.PDB.MMCIF2Dict import MMCIF2Dict
#from Bio.PDB.MMCIFParser import MMCIFParser
#from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import numpy as np
#import math
import sys
import time
import Bio
from module import *

t_dic={'ALA':'A','VAL':'V','LEU':'L','ILE':'I','PHE':'F','TRP':'W','MET':'M','PRO':'P',\
       'GLY':'G','SER':'S','THR':'T','CYS':'C','TYR':'Y','ASN':'N','GLN':'Q','HIS':'H',\
       'LYS':'K','ARG':'R','ASP':'D','GLU':'E'}
path = "/home/cxy/旧电脑/PycharmProjects/gsf/pdb_/"
pdb_list_file = "file/cullpdb_pc25_res2.0_R0.25_d181126_chains9311"


if __name__ == '__main__':
    p = PDBParser(PERMISSIVE=0)  
    pdb_id, pdb_chain = get_id_chain_name(pdb_list_file)
    for i in range(len(pdb_id)):
        if len(pdb_id[i]) !=4:
            continue
        pdb_name=path + "pdb"+pdb_id[i].lower()+'.ent'
        print(pdb_name)
        try:
            s = p.get_structure("1",pdb_name)       #read pdb struture
            s = s[0][pdb_chain[i]]                   #choose chain
            res_list = PDB.Selection.unfold_entities(s, 'R')   #read aminoacid
        except:
            print("read %s fail! " % pdb_name)
            continue
        aa_list = get_aa_list(res_list)
        aa_list_full = check_aa_id(aa_list)
        if not aa_list_full:
            print("aa_list error!")
            continue
        
        dps = cal_depth(s, aa_list_full)
        hse_a, hse_b = cal_hseab(s, aa_list_full)
        
        seq_list = get_seq(aa_list_full)
        ca_list = get_atom_list(aa_list_full,'CA')
        cb_list = get_atom_list(aa_list_full,'CB')
        c_list = get_atom_list(aa_list_full,'C')
        n_list = get_atom_list(aa_list_full,'N')

        ca_dist = cal_dist(ca_list)
        mask = get_mask(ca_list)            
        ids=ca_dist==None
        ca_dist[ids]=100   #算不出来距离的设置为100
        ca_dist_cs=[]
        angle_cs=[]
        num_cs=[]
        for j in range(len(ca_dist)):
            t = ca_dist[j]
            s=t.argsort()
            aa_num24 = s[1:25]
            ca_dist_cs.append(t[s[1:25]])
            angle_d = get_angle5_ceshi(aa_num24, ca_list, cb_list, n_list, c_list, j)
            angle_d = np.array(list(angle_d))
            angle_cs.append(angle_d)
            #angle_cs.append(angle_d[j][s[1:17]])
            #print(angle_d[j][s[1:17]])
            num_cs.append(s[1:25])

        dic_r={}
        dic_r['dis']=ca_dist_cs #距离
        dic_r['angle']=angle_cs #角度
        dic_r['mask']=mask      #标记ca原子，1有，0无
        dic_r['ids']=num_cs    # 氨基酸序号
        dic_r['seq']=seq_list  #序列
        dic_r['dps']=dps        #氨基酸深度
        dic_r['hsea']=hse_a     #裸球暴露面积
        dic_r['hseb']=hse_b
        out_name='pdb_other_cb/'+pdb_id[i].lower()+pdb_chain[i]+'_all_c.npy'
        np.save(out_name,dic_r)
        print("cal finish!")
