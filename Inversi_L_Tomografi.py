import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


# Membuat data traveltime sintetik dengan matriks yang ditentukan sendiri
S = np.array((1,1,1,1,0.2,1,1,1,1))    # data matriks slowness untuk matriks 3x3. 
                                                                    
# Membuat matriks kernel yang menampung nilai ray
G = [[1,1,1,0,0,0,0,0,0],       # ray melewati blok 1 2 3 
     [0,0,0,1,1,1,0,0,0],       # ray melewati blok 4 5 6 
     [0,0,0,0,0,0,1,1,1],       # ray melewati blok 7 8 9 
     [1,0,0,1,0,0,1,0,0],       # ray melewati blok 1 4 7 
     [0,1,0,0,1,0,0,1,0],       # ray melewati blok 2 5 8 
     [0,0,1,0,0,1,0,0,1],       # ray melewati blok 3 6 9 
     [np.sqrt(2),0,0,0,np.sqrt(2),0,0,0,np.sqrt(2)],        # ray melewati blok 1 5 7 
     [0,0,np.sqrt(2),0,np.sqrt(2),0,np.sqrt(2),0,0],        # ray melewati blok 3 5 7 
     [0,0,0,0,0,np.sqrt(2),0,np.sqrt(2),0]]         # ray melewati blok 6 8 

G=np.array((G))
# Menghitung traveltime dengan mengkalikan kernel dan slowness
dT = G.dot(S.T) # matriks traveltime

I = np.eye (9,9) # matriks indentitas 3x3

# Menentkan faktor redaman
L=np.ones((3))
E= np.ones((3))

# bobot 0
eps_1 = 1;
m_1= inv(((G.T).dot(G)) + ((eps_1**2)*I)).dot(G.T).dot(dT) # solusi slowness hasil inversi dengan bobot 0
E[0]=((dT-(G.dot(m_1))).T).dot(dT-G.dot(m_1))     #kriteria solusi misfit minimum
L[0]=(m_1.T).dot(m_1)       #kriteria solusi model minimum
                                         
# bobot 0.5
eps_2 = 2.5;
m_2= inv(((G.T).dot(G)) + ((eps_2**2)*I)).dot(G.T).dot(dT) # solusi slowness hasil inversi dengan bobot 0.5
E[1]=((dT-(G.dot(m_2))).T).dot(dT-G.dot(m_2))     #kriteria solusi misfit minimum
L[1]=(m_2.T).dot(m_2)       #kriteria solusi model minimum

# bobot 0.8
eps_3 = 3;
m_3= inv(((G.T).dot(G)) + ((eps_3**2)*I)).dot(G.T).dot(dT) # solusi slowness hasil inversi dengan bobot 0.8
E[2]=((dT-(G.dot(m_3))).T).dot(dT-G.dot(m_3))     #kriteria solusi misfit minimum
L[2]=(m_3.T).dot(m_3)       #kriteria solusi model minimum

# Memplot gambar

plt.figure(1)
ax1 = plt.imshow([ [1, 1, 1], [1, 0.2, 1], [1, 1, 1]])
plt.colorbar(ax1)

plt.figure(2)
ax2=plt.imshow([[m_1[6], m_1[7], m_1[8]], [m_1[3], m_1[4], m_1[5]], [m_1[0], m_1[1], m_1[2]]])
plt.colorbar(ax2)

plt.figure(3)
ax3 = plt.imshow([[m_2[6], m_2[7], m_2[8]], [m_2[3], m_2[4], m_2[5]], [m_2[0], m_2[1], m_2[2]]])
plt.colorbar(ax3)

plt.figure(4)
ax4 = plt.imshow([[m_3[6], m_3[7], m_3[8]], [m_3[3], m_3[4], m_3[5]], [m_3[0], m_3[1], m_3[2]]])
plt.colorbar(ax4)

plt.figure(5)
plt.plot (E,L)

plt.show()