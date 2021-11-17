import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

#Input data
x = np.array((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15))
y = np.array((3.5,5,6.5,8,9.5,11,12.5,14,15.5,17,18.5,20,21.5,23,24.5))

#Perhitungan regresi klasik untuk medapatkan solusi 
s = np.ones((2))

sum_x=0;
sum_y=0;
sum_x2=0;
sum_xy=0;

for i in range (0,len(x)):
    sum_x = x[i]+sum_x;
    sum_y = y[i]+sum_y;
    sum_x2 = (x[i]**2)+sum_x2;
    sum_xy = (x[i]*y[i])+sum_xy;

s[1] = ((len(x)*sum_xy)-(sum_x*sum_y))/((len(x)*sum_x2)-(sum_x**2))
s[0] = (sum_y/len(x))-(s[1]*(sum_x/len(x)))

#Matriks baru berisi pembulatan y
y_1=np.ones(len(x))
for i in range (0,len(x)):
    y_1[i]=round(y[i])

#mendefinisikan matriks kernel
G = np.ones((len(x),2))
x=x.T
y=y.T

for i in range (0,len(x)):
    G[(i,1)] = x[i]

#menghitung Parameter model solusi    
m = (inv((G.T).dot(G))).dot(G.T).dot(y_1)

y_2 = G.dot(m)

#Melakukan Perhitungan kovariansi Cm
C=(0.5**2)*(inv((G.T).dot(G)))
cm=np.zeros((len(m),len(m)))

for i in range (0,len(m)):
    for j in range (0, len(m)):
        if j==i:
            cm[(i,j)] = C[(i,j)]
            
delta_m = np.zeros(len(m))
m_upper = np.zeros(len(m))
m_lower = np.zeros(len(m))

for i in range (0, len(m)):
    delta_m[i]=cm[(i,i)]
    m_upper[i] = m[i] + delta_m[i]
    m_lower[i] = m[i] - delta_m[i]

#Melakukan plot grafik
xplot=np.array(range(0,len(x)+3))
yplot=np.ones(len(xplot))
yplot_upper=np.ones(len(xplot))
yplot_lower=np.ones(len(xplot))

for i in range (0,len(x)+3):
    yplot[i] = m[0]+xplot[i]*m[1]
    yplot_upper[i] = m_upper[0]+xplot[i]*m_upper[1]
    yplot_lower[i] = m_lower[0]+xplot[i]*m_lower[1]

plt.figure(figsize=(10,5))    
plt.plot(xplot, yplot, '-b', label='Regresi Linear')
plt.plot(xplot, yplot_upper, '--r')
plt.plot(xplot, yplot_lower, '--r')
plt.plot(x,y_1, 'og', label='Data')
plt.legend(loc='lower right')
plt.xlabel('X')
plt.ylabel('Y')


plt.show()
