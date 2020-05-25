from scipy import optimize
from sympy import *
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import os
import glob
import matplotlib as mpl
import mpl_scatter_density
from scipy.stats import gaussian_kde
plt.rcParams.update({'figure.max_open_warning': 0})
plt.style.use('classic')


######데이터 불러오는 부분#######
path = 'D:/PN_Noise_All/25T/G40' ###파일경로

file_list= os.listdir(path)

print(file_list)
print(path + '/'+file_list[0])

sort_list = []
search = 'm'
for i in range(0, len(file_list)):
    if file_list[i].find(search)>0:
        sort_list.append(np.float(file_list[i][:-2])*0.001)
    else:
        sort_list.append(np.float(file_list[i][:-1]))

print(sort_list)

s = np.argsort(sort_list)

final_sort_list = []
final_file_list = []
for i in range(0, len(s)):
    final_sort_list.append(sort_list[s[i]])
    final_file_list.append(file_list[s[i]])

print(final_file_list)
##############파일 정렬기능###############

frequency, si, Nsi, time, current = [], [], [], [], []

for i in range(0, len(final_file_list)):
    data = np.genfromtxt(path + '/' + final_file_list[i], delimiter='\t', skip_header=1, unpack=True)
    print(path + '/' + final_file_list[i])
    frequency.append(data[0][:15000])
    si.append(data[1][:15000])
    Nsi.append(data[2][:15000])
    time.append(data[3][:15000])
    current.append(data[4][:15000])

'''
np.savetxt(path+'/'+'frequency.csv', np.transpose(frequency), delimiter=',')

np.savetxt(path+'/'+'si.csv', np.transpose(si), delimiter=',')

np.savetxt(path+'/'+'Nsi.csv', np.transpose(Nsi), delimiter=',')

np.savetxt(path+'/'+'time.csv', np.transpose(time), delimiter=',')

np.savetxt(path+'/'+'current.csv', np.transpose(current), delimiter=',')
'''

def make_plot_each(x,y):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()
    ax.plot(x,y)
    return ax

def make_plot_one(x,y,z):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()
    for i in range(0, len(y)):
        ax.loglog(x,y[i], label = '{0}'.format(final_sort_list[i]) + 'V',color = colors[i])
    ax.grid(True,which='both')
    ax.legend()
    return ax

#ax.title.set_text('{0}'.format(z))
#######time_current plot##########
'''
for i in range(0, len(final_file_list)):
    make_plot_each(time[i],current[i])
'''
###########1/fnoise plot############
'''
n = len(final_file_list)
colors = pl.cm.jet(np.linspace(0,1,n))

make_plot_one(frequency[0],si,final_sort_list)
'''

#######trap_site_plot#########

def make_plot_trap(x):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x[1:], x[:-1])
    #cmap 써보자나중에
    fig.colorbar(density, label='number of points')
    return ax

#for i in range(0, np.int(len(current)/5)):

make_plot_trap(current[15])

'''
def make_plot_kde(x):
    xx = np.vstack([x[1:],x[:-1]])
    z = gaussian_kde(xx)(xx)
    idx = z.argsort()
    x,z = x[idx],z[idx]
    fig, ax = plt.subplots()
    ax.scatter(x,x,c=z,s=10,edgecolor = '')
    return ax

make_plot_kde(current[15])
'''
plt.show()
