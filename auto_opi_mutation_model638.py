import numpy as np
import Bio.SVDSuperimposer as SVD
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import copy
import simnetnb 
import sys
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import argparse
import math_p
from Bio.PDB.PDBParser import PDBParser
#import pdb_read.PDBParser as sc
from Bio import PDB
from Bio.PDB.vectors import calc_angle 
from Bio.PDB.vectors import calc_dihedral
from cal_pdb_feature_jiasu import *
#from rmsd import cal_rmsd
from pre_data import get_backbone, get_inner_coord, get_inner_coord_cb, check_backbone_3, get_cb_list, get_feature_matrix
from simnetnb import NeRF_net_cb, NeRF_net, g_data_net

seqdic={'A':0, 'R':1, 'D':2, 'C':3, 'Q':4, 'E':5, 'H':6, 'I':7, 'G':8, 'N':9, 'L':10, 'K':11, 
        'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19}


def save_opi_strcture(c,c2,aa_list_full,j, path):
    res_num= 0
    filename = (path+"/%s.pdb" % (j+1))
    serial_num = 0
    chain = 'A'
    with open (filename,'w') as outfile:    
        for i, res in enumerate(aa_list_full):
            res_name = res.resname
            res_num = res_num+1
            for j in range(3):
                serial_num= serial_num +1
                x,y,z = c[3*i+j]
                if j == 0:
                    atom_name = 'N'
                if j == 1:
                    atom_name = 'CA'
                if j == 2:
                    atom_name = 'C'
                last2 = 1
                last1 = 0
                outfile.write( "ATOM%7d%5s%4s%2s%4d%12.3f%8.3f%8.3f%6.2f%6.2f\n" % 
                (serial_num, atom_name, res_name ,chain, res_num, x, y, z,last1, last2))
            serial_num= serial_num +1
            x,y,z = c2[i]
            atom_name = 'CB'
            last2 = 1
            last1 = 0
            outfile.write( "ATOM%7d%5s%4s%2s%4d%12.3f%8.3f%8.3f%6.2f%6.2f\n" % 
            (serial_num, atom_name, res_name ,chain, res_num, x, y, z, last1, last2))


def path_opi_structure_pdb(PATH_R, decoy_name):
    ''' get save opi_structure pdb path '''
    PATH1 = PATH_R +"/"+decoy_name+'_'
    os.makedirs(PATH1, exist_ok=True)
    return PATH1

def get_decoy_list(path, native):
    f_name=os.listdir(path)
    pdb_list = []
    for name in f_name:
        if name != native:
            pdb_list.append(name)
    return pdb_list

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        n = 0
        for param in group["params"]:
            if param.grad is not None:
                #n = n+1
                print(param.grad.data,n)
                param.grad.data.clamp_(-grad_clip, grad_clip)

def read_pdb(pdb_id):
    p = PDBParser(PERMISSIVE=1)
    s = p.get_structure("1",pdb_id) 
    #print("***%s***" % name)    
    s = s[0] 
    res_list = PDB.Selection.unfold_entities(s, 'R')   #read aminoacid
    aa_list = get_aa_list(res_list)
    aa_list_full = check_aa_id(aa_list)
    return aa_list_full

def cal_rmsd_ca(decoy_name, native):
    decoy_list = read_pdb(decoy_name)
    native_list = read_pdb(native)
    ca1 = []
    ca2 = []
    if len(decoy_list) != len(native_list):
        num1 = decoy_list[0].id[1]
        num2 = native_list[0].id[1]
        max_num = max([num1,num2])
        length_seq = len(decoy_list[max_num:])
    else:
        length_seq = len(decoy_list)
    for i in range(length_seq):
        if decoy_list[i] and native_list[i]:
            ca1.append(decoy_list[i]['CA'].coord)
            ca2.append(native_list[i]['CA'].coord)
    x = np.array(ca1)
    y = np.array(ca2)
    sup = SVD.SVDSuperimposer()
    sup.set(x, y)
    sup.run()
    rms = sup.get_rms()
    return rms

def ab_index(idx):
    h, w = idx.size()
    a = torch.arange(h).view(-1, 1)
    b = torch.ones(w).long().view(1, -1)
    ab = torch.mm(a, b)
    return ab

def entropy(outputs_d):
    pi = F.softmax(outputs_d,dim=1)
    log_pi = F.log_softmax(outputs_d,dim=1)
    entropy = torch.sum(-pi.mul(log_pi), 1)
    weight = 1-entropy/np.log(21)
    return weight

def get_l1_entropy_weight(entropy_weight, first_param):
    new_weight = entropy_weight.unsqueeze(0).repeat(len(y),1)[first_param[1], first_param[0]]
    return new_weight

def g_data(mask, num_cs, seq_list):
    ids=num_cs
    seq=seq_list
    idx_nb = []
    idx_unb = []
    label = []
    index_h=[]
    index_nb =[]
    index_unb = []
    kk=0
    for i in range(len(mask)):
        nb_id = [j for j in range(-6+i,7+i) if j!=i]
        if mask[i]!=0:
            index_h.append(i)
            idx_nb_i = []
            idx_unb_i = []
            index_nb_i = []
            index_unb_i = []
            for j in range(len(ids[i])):
                if len(idx_unb_i)==10:
                    break
                if abs(ids[i][j]-i)>6:
                    index_unb_i.append(j)
                    if seq[ids[i][j]] in seqdic:
                        idx_unb_i.append(seqdic[seq[ids[i][j]]])
                    else:
                        idx_unb_i.append(20)
            while len(idx_unb_i) < 10:
                index_unb_i.append(-1)
                idx_unb_i.append(21)
            for a in nb_id:
                if a in ids[i]:
                    k=np.where(ids[i]==a)
                    k=int(k[0][0]) #duole yiwei suoyixuyao suoyin liangci
                    index_nb_i.append(k)
                    if seq[a] in seqdic:
                        idx_nb_i.append(seqdic[seq[a]])
                    else:
                        idx_nb_i.append(20)
                else:
                    index_nb_i.append(-1)
                    idx_nb_i.append(21)
            idx_nb.append(idx_nb_i)
            idx_unb.append(idx_unb_i)
            index_nb.append(index_nb_i)
            index_unb.append(index_unb_i)
            if seq[i] in seqdic:
                label.append(seqdic[seq[i]])
            else:
                label.append(20)
            kk+=1
    index_unb = torch.tensor(index_unb)
    index_nb1 = torch.tensor(index_nb)
    idx_nb = torch.tensor(idx_nb)
    idx_unb = torch.tensor(idx_unb)
    index = torch.cat((index_unb,index_nb1), 1)
    idx = torch.cat((idx_unb, idx_nb),1)
    del index_nb1
    label1 = torch.tensor(label)
    index_h1 = torch.tensor(index_h)
    return  idx, label1, kk, index, index_h1


if __name__ == '__main__':
    #########parameter###########
    #python auto_opi_mutation_model660.py --PATH opi_pdb_structure/Dfine/native_start --native_name 1a00A_native.pdb --decoy_name 1a00A_model.pdb --cuda cpu --LR 0.0002
    parser = argparse.ArgumentParser()
    parser.add_argument('--ITER', type=int, default=6)
    parser.add_argument('--WINDOW', type=int, default=16)
    parser.add_argument('--LR', type=float, default=0.001)
    parser.add_argument('--ENRAOPY_W', type=int, default=0)
    parser.add_argument('--L1_smooth_parameter', type=float, default=0.0)
    parser.add_argument('--PATH',  default="native_start")
    parser.add_argument('--native_name',  default="1a00A_native.pdb")
    parser.add_argument('--decoy_name', default="1a00A_model.pdb")
    parser.add_argument('--device', default = "cpu")
    parser.add_argument('--OPI', default = "SGD")
    opt = parser.parse_args()

    ####################################
    #force fileld and model
    ####################################
    #pth_file='modelnb/best_seq_dist_angle_sincos16aa_tian0_gpu.pth'
    #device =torch.device(opt.cuda if torch.cuda.is_available() else "cpu")
    device = torch.device(opt.device)
    pth_file='modelnb/best_seq_dis_angle_638wei.pth'
    model = simnetnb.Sim()
    model.load_state_dict(torch.load(pth_file, map_location='cpu'))
    result_path = "660_lr"+str(opt.LR)+"_sm"+str(opt.L1_smooth_parameter)+opt.OPI+"_entropy"+str(opt.ENRAOPY_W) 
    PATH_R = opt.PATH+"/"+result_path     #decoy generate path  
    native = opt.PATH+'/'+opt.native_name #native structure path
    g_data=g_data_net()
    path_decoy = path_opi_structure_pdb(PATH_R ,opt.decoy_name) #
    decoy_id = opt.PATH +"/"+opt.decoy_name  # opi decoy pdb

    ####################################
    ###read pdb_file and cal inner coord
    ####################################
    aa_list_full = read_pdb(decoy_id) 
    ca_list, c_list, n_list = check_backbone_3(aa_list_full)
    cb_list = get_cb_list(aa_list_full)
    backbone  = get_backbone(ca_list, c_list, n_list)
    mask = get_mask(ca_list)
    seq_list = get_seq(aa_list_full) 
    inner_tensor = get_inner_coord(backbone)  #c coord to inner coord and cal feature 
    inner_coord_cb_tensor = get_inner_coord_cb(ca_list, cb_list, n_list, c_list)
    mainchain_coord_tensor = torch.from_numpy(np.array([i.coord for i in backbone]))
    net = NeRF_net(inner_tensor)  #model
    net2 = NeRF_net_cb(inner_coord_cb_tensor)     
    criterion = nn.NLLLoss()
    criterion_none = nn.NLLLoss(reduction = 'none') #loss not tp be mean or sum
    criterion_SmoothL1Loss = torch.nn.SmoothL1Loss()
    #optimizer = optim.SGD(net.dhd_v, lr=LR)
    if opt.OPI == "ADAM":
        optimizer = optim.Adam(net.dhd_v, lr=opt.LR)
    else:
        optimizer = optim.SGD(net.dhd_v, lr=opt.LR)
    
    ###################################
    ##############train################
    ###################################
    #print("*"*25,opt.decoy_name,"lr:", opt.LR,"sml1:",opt.L1_smooth_parameter,"*"*25, )
    rmsd_file = path_decoy + "/rmsd.txt"
    first_param = []
    dist_list = []
    with open (rmsd_file,'w') as t:
        for i in range(opt.ITER):
            optimizer.zero_grad()
            c = net(mainchain_coord_tensor) #generate backbone coordiante
            c2 = net2(c)                    #generate cb coordiante
            dist, angle, num_cs, ca_dist, ab = get_feature_matrix(c,c2,opt.WINDOW)
            save_opi_strcture(c,c2,aa_list_full,i,path_decoy)
            x, idx_, dis_, angle_t, y, kkn = g_data(mask, num_cs , dist, angle, seq_list)
            outputs=model(x,idx_,dis_,angle_t)
            _, preds = torch.max(outputs, 1)
            running_corrects = 0
            running_corrects += torch.sum(preds == y.data)
            acc = running_corrects.double().cpu().data.numpy()/len(y)

            ##############change loss by acc####################### 
            #loss = (criterion(F.log_softmax(outputs,dim=1), y))*(acc**2)*10.0 #gengju acc chengfa xunlian de haobuhao de 

            ##############change loss by predicted right aa########
            #x = (preds == y.data)
            #loss = criterion(F.log_softmax(outputs[x],dim=1), y[x])

            ###############normal loss #############################           
            #loss = criterion(F.log_softmax(outputs,dim=1), y)

            #######paramter about distance restrain loss##########
            if opt.L1_smooth_parameter:
                if i == 0: #i =0 is first iter, Get the index of 16 amino acids around the target amino acid
                    first_param.append(num_cs.cpu().detach()) #Sencond dimension index of surrounding 16 amino acids eg:[5,13,16,20...]
                    first_param.append(ab.cpu().detach()) #First dimension index of surrounding 16 amino acids eg:[0,0,0,...,0]
                    dist_list.append(dist.cpu().detach())
                ######distance restrain loss#####
                #l2_loss = 0.002*torch.norm(dist,2)
                dist1 = dist_list[0]
                distn = ca_dist[first_param[1], first_param[0]]
                dist_deta = distn - dist1
                l2_loss = criterion_SmoothL1Loss(dist_deta, torch.zeros(len(y), opt.WINDOW)) 
                #l2_loss = criterion_SmoothL1Loss(dist_deta*entropy_weight_l2, torch.zeros(len(y), WINDOW))
            else:
                l2_loss = 0
            ########entropy weight restrain loss#################
            if opt.ENRAOPY_W:
                outputs_d = outputs.detach()
                entropy_weight = entropy(outputs_d) #
                entropy_weight_l2 = get_l1_entropy_weight(entropy_weight, first_param) 
                loss = torch.mean(criterion_none(F.log_softmax(outputs,dim=1), y)*entropy_weight) # a entroy weight to every aa site
            else:
                loss = criterion(F.log_softmax(outputs,dim=1), y)

            loss_all = opt.L1_smooth_parameter*l2_loss + loss 

            ###########calculate RMSD and DGT_HA_SCORE#######################
            start_model = (path_decoy+"/%d.pdb" % (i+1))
            tm_score = os.popen("java -jar TM/TMscore.jar %s %s\n" % (start_model,native))
            try:
                tm = tm_score.read().split('\n')[17]
            except:
                tm = tm_score.read().split('\n')
            RMSD = cal_rmsd_ca(start_model,native)          

            #print("epoch:%d" % i)
            print("acc:%.4f,   loss_all:%.4f, RMSD:%.4f, %s\n" % (acc*100, loss_all.item(), RMSD, tm[:20]))
            #print("%s" % tm)
            #t.write("acc:%.4f loss_all:%.4f, RMSD:%.4f, %s\n" % (acc*100, loss_all.item(), RMSD, tm[:20]))
            loss_all.backward()
            #clip_gradient(optimizer, 0.01)
            optimizer.step()
    rm_path = path_decoy+"/*.pdb"
    os.system("rm %s" % rm_path)

                



