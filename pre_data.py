import torch
import numpy as np
import math_p as mp
import random
from random import choice
from cal_pdb_feature_jiasu import *
#np.random.seed(1)
#random.seed(1)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#window = 16
def get_cs_num(ca_dist,window=16):
    """ 
        output: aa num around target aa sorted by distance
        ca_dist: n*n tensor , n is sequence length
        window: cutoff of the round aa, defalt 16
    """
    return ca_dist.argsort()[:,1:window+1]

def get_backbone(ca_list,c_list, n_list):
    """
        oputput: mainchain atom
        ca_list: Ca atom of all amino acids
        c_list: c atom of all amino acids
        n_list: n atom of all amino acids
    """
    mainchain = []
    for i in range(len(ca_list)):
        mainchain.append(n_list[i])
        mainchain.append(ca_list[i])
        mainchain.append(c_list[i])
    return mainchain

def get_inner_coord(mainchain):
    
    inner = []
    for i in range(3,len(mainchain)):
        atom = mainchain[i].coord
        #print(mainchain[i].id,atom)
        nb1 = mainchain[i-1].coord
        nb2 = mainchain[i-2].coord
        nb3 = mainchain[i-3].coord
        vec1 = nb2- nb3
        vec2 = nb1- nb2
        vec3 = atom- nb1
        #print(vec1.dtype,vec2.dtype,vec3.dtype)
        bond = mp.L_MO_ab(atom, nb1)
        #print(bond.dtype)
        angle = mp.get_angle(vec2, vec3)
        #print("angle",angle.dtype)
        dhd = mp.get_dhd(vec1, vec2, vec3)
        #print(dhd.dtype)
        inner_coord = bond, angle, dhd
        #print(inner_coord)
        inner.append(inner_coord)
    a = torch.from_numpy(np.array(inner,dtype= 'float32'))
    return a

def get_inner_coord_cb(ca_list, cb_list, n_list, c_list):
    inner = []
    for i in range(len(cb_list)):
        nb3 = n_list[i].coord
        nb2 = ca_list[i].coord
        nb1 = c_list[i].coord
        #print(cb_list[i], c_list[i], n_list[i], ca_list[i])
        if cb_list[i]==None and c_list[i]!=None and n_list[i]!=None and ca_list[i]!=None:
            #print(1)
            ca_v=ca_list[i].get_vector().get_array()
            c_v=c_list[i].get_vector().get_array()
            n_v=n_list[i].get_vector().get_array()
            cb=calha1(n_v,c_v,ca_v)
            #cb=PDB.vectors.Vector(cb)
            atom= cb
        else:
            atom = cb_list[i].coord
        
        vec1 = nb2- nb3
        vec2 = nb1- nb2
        vec3 = atom- nb1
        #print(vec1,vec2,vec3)
        bond = mp.L_MO_ab(atom, nb1)
        #print(bond)
        angle = mp.get_angle(vec2, vec3)
        dhd = mp.get_dhd(vec1, vec2, vec3)
        inner_coord = bond, angle, dhd
        #print(inner_coord)
        inner.append(inner_coord)
    a = torch.from_numpy(np.array(inner,dtype= 'float32'))
    return a

def get_angle(a,b): 
    """
        calculate two tensor angle batch
        a(x,y,z), b(x,y,z)
    """  
    c = torch.dot(a,b)
    aa = torch.norm(a)
    bb = torch.norm(b)
    tmp = c/(aa*bb)
    if tmp > 1.0:
        tmp = 1.0
    if tmp < -1.0:
        tmp = -1.0
    theta = torch.squeeze(torch.Tensor([np.pi]),0)-torch.acos(tmp)
    return theta#

def get_angle_matrix(a, b):
    c = a * b  #16*3
    #print(a,b)
    c_sum = torch.sum(c, 1)
    #print("c_sum", c_sum)
    #c_u = torch.squeeze(c_sum,1)
    #print(c_u.shape)
    aa = a * a
    bb = b * b
    aa_norm = torch.sum(aa, 1)
    bb_norm = torch.sum(bb, 1)
    #print(bb_norm)
    ab = torch.rsqrt(aa_norm * bb_norm)
    tmp = c_sum * ab
    theta = torch.Tensor([np.pi]).to(device) - torch.acos(tmp)
    return theta

def cal_ABdistance(A, B):
    "calculate ca distance by matrix"
    #print("A",A)
    #print("B",B)
    B_t = torch.t(B)
    vecProd = torch.mm(A, B_t)
    SqA = A**2
    SqB = B**2
    #print("SqA:", SqA)
    #print("SqA:", SqB)
    sumSqA = torch.sum(SqA, 1)
    sumSqB = torch.sum(SqB, 1)
    a_len = len(sumSqA)
    b_len = len(sumSqB)
    sumSqA_w = sumSqA.view(1, -1)
    sumSqB_w = sumSqB.view(1, -1)
    sumSqBx = torch.t(sumSqB_w)
    eye = torch.ones((a_len,1),device = device)
    eye1 = torch.ones((1,a_len), device = device)
    sumSqAx = torch.mm(eye,sumSqA_w)
    sumSqBx_t = torch.mm(sumSqBx,eye1)
    SqED = sumSqAx + sumSqBx_t - 2 * vecProd
    #print(SqED[:10])
    SqED_0 =torch.where(SqED<0, torch.zeros(1,device = device), SqED)
    SqED_sq = torch.sqrt(SqED_0+1e-8)
    return SqED_sq

def get_angle6_matrix(num_cs, coord_ca, coord_cb, coord_n, coord_c):
    """calculate angle by matrix
       
    """
    num_cs_1 = num_cs.contiguous().view(-1)
    h,w = num_cs.size()
    ca1 = coord_ca.unsqueeze(1)   #100*3
    ca2 = coord_ca[num_cs_1].view(-1,w,3) #100*16*3
    cb1 = coord_cb.unsqueeze(1)
    cb2 = coord_cb[num_cs_1].view(-1,w,3)
    n1 = coord_n.unsqueeze(1)
    n2 = coord_n[num_cs_1].view(-1,w,3)
    c1 = coord_c.unsqueeze(1)
    c2 = coord_c[num_cs_1].view(-1,w,3)
    eye = torch.ones(w, 1)
    #print(ca2.shape, ca1.shape )
    ca1_cb1 = (ca1 - cb1).repeat(1,w,1).view(-1,3) #100*3
    ca1_n1 = (ca1 - n1).repeat(1,w,1).view(-1,3)
    ca1_c1 = (ca1 - c1).repeat(1,w,1).view(-1,3)
    ca2_ca1 = (ca2 - ca1).view(-1,3)
    ca2_cb2 = (ca2 - cb2).view(-1,3)
    ca2_n2 = (ca2 - n2).view(-1,3)
    ca2_c2 = (ca2 - c2).view(-1,3)
    t1 = get_angle_matrix(ca1_cb1, ca2_ca1)
    t11 = get_angle_matrix(ca2_ca1, -ca2_cb2)
    t2 = get_angle_matrix(ca1_n1, ca2_ca1)
    t22 = get_angle_matrix(ca2_n2, -ca2_ca1)
    t3 = get_angle_matrix(ca1_c1, ca2_ca1)
    t33 = get_angle_matrix(ca2_c2, -ca2_ca1)
    angle = [t1, t2, t3, t11, t22, t33]
    angle_t = torch.t(torch.cat(angle).view(6, -1))
    yield angle_t

def get_feature_matrix(c, c2, window=16):
    coord_n = c[::3].clone()
    coord_ca = c[1::3].clone()
    coord_c = c[2::3].clone()
    coord_cb = c2
    A = coord_ca.clone()
    B = coord_ca.clone()
    aa_num = len(A)
    ca_dist = cal_ABdistance(A, B)
    num_cs = get_cs_num(ca_dist,window)
    ab = ab_index(num_cs)
    ab = ab.to(device)
    ca_dist_new = ca_dist[ab, num_cs]
    angle_d_m = get_angle6_matrix(num_cs, coord_ca, coord_cb, coord_n, coord_c)
    angle_d_m = list(angle_d_m)[0]
    angle_d_m = angle_d_m.view(-1, window, 6)
    return ca_dist_new, angle_d_m, num_cs, ca_dist, ab


def read_pdb(pdb_id, chain):
    p = PDBParser(PERMISSIVE=1)
    s = p.get_structure("1", pdb_id)
    #print("***%s***" % name)
    s = s[0][chain]
    res_list = PDB.Selection.unfold_entities(s, 'R')  #read aminoacid
    aa_list = get_aa_list(res_list)
    aa_list_full = check_aa_id(aa_list)
    return aa_list_full


def get_pdb_name(PATH, train_name):
    pdb_id = train_name[:4]
    chain = train_name[4]
    pdb_name = PATH + "pdb" + pdb_id.lower() + '.ent'
    return pdb_name, chain


def cal_ca_dist(ca_list):
    ca_num = len(ca_list)
    ca_dist = np.array([
        ca_list[i] - ca_list[j] for i in range(ca_num) for j in range(ca_num)
    ])
    ca_dist = ca_dist.reshape(-1, ca_num)
    return ca_dist


def get_atom_list(aa_list_full, atom):
    """parameters:
       aa_list_full:all the residues class
       atom: the name of atom,"CA","C","N"
    """
    atom_list = []
    for i, a in enumerate(aa_list_full):
        try:
            t = a[atom]
            atom_list.append(t)
        except:
            print("res %d not have atom %s" % (i+aa_list_full[0].id[1],atom))
            sys.exit()
            #continue
    return atom_list


def get_cb_list(aa_list_full):
    cb_list = []
    for a in aa_list_full:
        try:
            t = a['CB']
            cb_list.append(t)
        except:
            cb_list.append(None)
    return cb_list


def check_backbone_3(aa_list_full):
    """
        Checking the main chain for lack of atoms
    """
    ca_list = get_atom_list(aa_list_full, 'CA')
    c_list = get_atom_list(aa_list_full, 'C')
    n_list = get_atom_list(aa_list_full, 'N')
    if ca_list and c_list and n_list:
        return ca_list, c_list, n_list
    else:
        return 0, 0, 0


def mutation_dic_set(ca_dist, DIS_CUTOFF):
    ca_dist = np.round(ca_dist,2)
    dic_muset = {}
    for i in range(len(ca_dist)):
        non_num =np.where(ca_dist[i]>DIS_CUTOFF)
        #print(non_num[0])
        dic_muset[i] = non_num[0]
    return dic_muset

def get_first_mutation_site(dic_muset,ca_dist):
    mu1 = np.random.randint(0,len(ca_dist))
    set1 = set(dic_muset[mu1])
    return mu1, set1

def get_mu_num(dic_muset, mu1, set1):
    mutation_list=[]
    mutation_list.append(mu1)
    if not list(set1):
        return mutation_list
    else:        
        mu2 = choice(list(set1))
        set2 = dic_muset[mu2]
        set12 = set(set1.intersection(set2))
        return get_mu_num(dic_muset, mu2, set12)

def ab_index(idx):
    h, w = idx.size()
    a = torch.arange(h).view(-1, 1)
    b = torch.ones(w).long().view(1, -1)
    ab = torch.mm(a, b)
    return ab


def get_angle5_ceshi(aa_num16, ca_list, cb_list, n_list, c_list, j):
    angle_t = []
    #cb1 = get_cb_vector(ca_list, cb_list, n_list, c_list, j)                
    for k in aa_num16:
        try:           
            ca1 = ca_list[j]
            cb1 = cb_list[j]
            ca2 = ca_list[k]
            vec1 = ca1-cb1
            vec2 = ca2-ca1
            #print(vec1)
            t1 = get_angle(vec1, vec2)
            t1 = torch.unsqueeze(t1,0)
        except:
            t1 = torch.Tensor([1])       
        try:
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            cb2 = cb_list[k]
            vec1 = ca2-ca1
            vec2 = cb2-ca2
            t11 = get_angle(vec1, vec2)
            t11 = torch.unsqueeze(t11,0)
        except:
            t11 = torch.Tensor([1])
        try:            
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            n_1 = n_list[j]
            vec1 = ca1-n_1
            vec2 = ca2-ca1
            t2 = get_angle(vec1, vec2)
            t2 = torch.unsqueeze(t2,0)
        except:
            t2 = torch.Tensor([1])
        try:
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            n_2 = n_list[k]
            vec1 = ca2-n_2
            vec2 = ca1-ca2
            t22 = get_angle(vec1, vec2)
            t22 = torch.unsqueeze(t22,0)
        except:
            t22 = torch.Tensor([1])
               
        try:
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            c_1 = c_list[j]
            vec1 = ca1-c_1
            vec2 = ca2-ca1
            t3 = get_angle(vec1, vec2)
            t3 = torch.unsqueeze(t3,0)
        except:
            t3 = torch.Tensor([1])

        try:
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            c_2 = c_list[k]
            vec1 = ca2-c_2
            vec2 = ca1-ca2
            t33 = get_angle(vec1, vec2)
            t33 = torch.unsqueeze(t33,0)
        except:
            t33 = torch.Tensor([1])
                
        angle_t = [t1,t2,t3,t11,t22,t33]
        #print(angle_t)
        angle_t = torch.cat(angle_t)
        yield angle_t

def get_feature(c,c2):
    """
        calculate feature by for 
    """
    n_list = [c[i] for i in range(0, len(c),3)]
    ca_list = [c[i+1] for i in range(0, len(c),3)]
    c_list = [c[i+2] for i in range(0, len(c),3)]
    cb_list = c2
    ca_dist = []
    ca_num = len(ca_list)
    for j in range(ca_num):
        for k in range(ca_num):
            ca_ca = torch.norm(ca_list[j]-ca_list[k],2) 
            ca_ca = torch.unsqueeze(ca_ca,0)               
            ca_dist.append(ca_ca)
    ca_dist=torch.cat(ca_dist)
    ca_dist=ca_dist.view(ca_num,ca_num)
    ca_dist_cs=[]
    angle_cs=[]
    num_cs=[]
    #print("get_feature pre cost:%fs" % time_cost(start)) 
    for j in range(len(ca_dist)):
        t = ca_dist[j]
        s=t.argsort()
        aa_num16 = s[1:17]
        ca_dist_cs.append(t[s[1:17]])
        angle_d = get_angle5_ceshi(aa_num16, ca_list, cb_list, n_list, c_list, j)
        angle_d = list(angle_d)
        angle_cs.append(angle_d)
        num_cs.append(s[1:17])
    #print("get_feature gen cost:%fs" % time_cost(start)) 
    return ca_dist_cs, angle_cs, num_cs


def get_angle6_fm(aa_num16, coord_ca, coord_cb, coord_n, coord_c, j):
    """
        calculte angle by for and matrix
    """
    ca1 = coord_ca[j]  #1*3
    ca2 = coord_ca[aa_num16]  #16*3
    cb1 = coord_cb[j]
    cb2 = coord_cb[aa_num16]
    n1 = coord_n[j]
    n2 = coord_n[aa_num16]
    c1 = coord_c[j]
    c2 = coord_c[aa_num16]
    eye = torch.ones((len(aa_num16), 1))
    ca1_cb1 = ca1 - cb1
    ca1_n1 = ca1 - n1
    ca1_c1 = ca1 - c1
    ca2_ca1 = ca2 - ca1  
    ca2_cb2 = ca2 - cb2
    ca2_n2 = ca2 - n2
    ca2_c2 = ca2 - c2        
    #vec1 = ca1_cb1.unsqueeze(0)  #1*3
    ca1_cb1_eye= torch.mm(eye, ca1_cb1.unsqueeze(0)) #16*3
    ca1_n1_eye= torch.mm(eye, ca1_n1.unsqueeze(0))
    ca1_c1_eye= torch.mm(eye, ca1_c1.unsqueeze(0))    
    t1 = get_angle_matrix(ca1_cb1_eye, ca2_ca1)
    #print(t1)
    t11 = get_angle_matrix(ca2_ca1, -ca2_cb2)
    t2 = get_angle_matrix(ca1_n1_eye, ca2_ca1)
    t22 = get_angle_matrix(ca2_n2 , -ca2_ca1)
    t3 = get_angle_matrix(ca1_c1_eye, ca2_ca1)
    t33 = get_angle_matrix(ca2_c2, -ca2_ca1)
    angle = [t1, t2, t3, t11, t22, t33]
    angle_t = torch.t(torch.cat(angle).view(6,-1))
    #print(t2)
    yield angle_t

def get_feature_fm(c,c2):
    coord_n = c[::3].clone()
    coord_ca = c[1::3].clone()
    coord_c = c[2::3].clone()
    coord_cb = c2
    A = coord_ca.clone()
    B = coord_ca.clone()
    aa_num = len(A)
    ca_dist = cal_ABdistance(A, B)
    num_cs = get_cs_num(ca_dist)
    #print("get_feature pre cost:%fs" % time_cost(start)) 
    angle_cs=[]
    ca_dist_cs = []
    for j in range(aa_num):
        aa_num16 = num_cs[j]
        t = ca_dist[j][aa_num16]
        ca_dist_cs.append(t)
        angle_d = get_angle6_fm(aa_num16, coord_ca , coord_cb, coord_n, coord_c, j)
        angle_d = list(angle_d)
        angle_cs.append(angle_d[0])
    #print("get_feature gen cost:%fs" % time_cost(start)) 
    return ca_dist_cs, angle_cs, num_cs
