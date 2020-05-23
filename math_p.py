#!/usr/bin/env python
import numpy as np
#from numba import jit

def L_cxyz (a, b):
	cx = a[1] * b[2] - b[1] * a[2]
	cy = a[2] * b[0] - b[2] * a[0]
	cz = a[0] * b[1] - b[0] * a[1]
	return cx, cy, cz#

def L_L_ab(a,b):
        c = (a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2 # 
        return c

#@nb.vectorize("float32(float32, float32)", nopython=True)
#@jit
def L_MO_ab(a,b):
	c = np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)
	return np.array(c,dtype = 'float32')
        #return np.linalg.norm(a-b) # 2范数
def rotation(r, v, theta):
    #r1=[]
    #t1 = []
    #t2 = []
    #t3 = []
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    t1 = np.dot(r, c_theta)
    t2 = L_cxyz(v,r)
    t2 = np.dot(t2, s_theta)
    vr = np.dot(v,r)
    t3 = (1 - c_theta) * np.dot(vr, v)
    r1 = t1 + t2 + t3
    return r1

def rotation_a(r0, r1, r2, n, v, theta):  #####
	t1 = np.zeros((3), dtype = np.float)
	t2 = np.zeros((3), dtype = np.float)
	t3 = np.zeros((3), dtype = np.float)
	vr = np.zeros((3), dtype = np.float)
	r = []
	c_theta = np.cos(theta)
	s_theta = np.sin(theta)
	for i in xrange(n):
		t1[0] = np.dot(r0[i], c_theta)
		t1[1] = np.dot(r1[i], c_theta)
		t1[2] = np.dot(r1[i], c_theta)
		t2 = L_cxyz(t2,s_theta)
		r = [r0[i], r1[i], r2[i]]
		vr = np.dot(r,v)
		t3 = np.dot(vr, (1 - c_theta)*v)
		r0[i] = t1[0] + t2[0] + t3[0]
		r1[i] = t1[1] + t2[1] + t3[1]
		r2[i] = t1[2] + t2[2] + t3[2]
	return r0, r1, r2	

def get_angle(a,b):   #两向量间角度ok
	c = np.dot(a,b)
	aa = np.linalg.norm(a)
	bb = np.linalg.norm(b)
	tmp = c/(aa*bb)
	if tmp > 1.0:
		tmp = 1.0
	if tmp < -1.0:
		tmp = -1.0
	theta = np.arccos(tmp)
	return theta#

def get_dhd(a, b, c):
	#fab = []
	#fbc = []
	#ff = []
	fab = L_cxyz(a,b)
	fbc = L_cxyz(b,c)
	m_fab = np.linalg.norm(fab)
	m_fbc = np.linalg.norm(fbc)
	d = np.dot(fab,fbc)
	tmp = d/(m_fab * m_fbc)
	if tmp > 1.0:
	    tmp = 1.0
	if tmp < -1.0:
	    tmp = -1.0
	angle = np.arccos(tmp)
	ff = L_cxyz(fab,fbc)
	if np.dot(ff,b) < 0:
	    angle = -angle
	return angle
