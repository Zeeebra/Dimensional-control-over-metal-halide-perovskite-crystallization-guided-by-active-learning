# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:40:16 2020

@author: Zhi Li
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def plot3d2d (point, x_range = [0, 2], y_range = [0, 5], z_range = [0, 10.5], xy_loc = -3, xz_loc = 0.1, yz_loc = -0.5,\
              x_step = 0.5, y_step = 1, z_step = 2, elev = 45, azim = -70, name = '_'):
    xy_plane = z_range[0] + xy_loc
    xz_plane = y_range[1] + xz_loc
    yz_plane = x_range[0] + yz_loc
    # 3D plot
    color_type = ['cornflowerblue', 'yellowgreen', 'r','orange','black']
    color_type_2D = ['cornflowerblue', 'yellowgreen','r','orange','black']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(1,6):
        s = 10 if i == 4 else 10
        ax.scatter(point[:,0][point[:,3]==i], point[:,1][point[:,3]==i], point[:,2][point[:,3]==i],\
                   c = color_type[i-1], s = s, alpha=0.5)   
    # 2D projections plot
    for i in range(1,6):
        s = 10 if i == 4 else 10
        arf = 0.3 if i == 4 else 0.3
        ax.scatter(point[:,0][point[:,3]==i],point[:,2][point[:,3]==i], \
                   marker = 'o', c = color_type_2D[i-1], s = s, zdir = 'y', zs = xz_plane, alpha=arf)
        ax.scatter(point[:,1][point[:,3]==i], point[:,2][point[:,3]==i], \
                   marker = 'o', c = color_type_2D[i-1], s = s, zdir = 'x', zs = yz_plane, alpha=arf)
        ax.scatter(point[:,0][point[:,3]==i], point[:,1][point[:,3]==i], \
                   marker = 'o', c = color_type_2D[i-1], s = s, zdir='z', zs = xy_plane, alpha=arf)
    # make 2D planes with meshgrid
    # yz plane
    yy, zz = np.meshgrid(np.arange(y_range[0],y_range[1]+y_step, y_step), np.arange(z_range[0],z_range[1]+z_step,z_step))
    xx = np.ones((len(np.arange(z_range[0],z_range[1]+z_step,z_step)), len(np.arange(y_range[0],y_range[1]+y_step, y_step))))*yz_plane
    ax.plot_surface(xx,yy,zz, color = "silver", alpha = 0.1)
    ax.plot_wireframe(xx,yy,zz, color = "black", alpha = 0.1)
    # xy plane
    xx, yy = np.meshgrid(np.arange(x_range[0],x_range[1]+x_step, x_step), np.arange(y_range[0],y_range[1]+y_step, y_step))
    zz = np.ones((len(np.arange(y_range[0],y_range[1]+y_step, y_step)), len(np.arange(x_range[0],x_range[1]+x_step, x_step))))*xy_plane
    ax.plot_surface(xx,yy,zz, color = "silver", alpha = 0.1)
    ax.plot_wireframe(xx,yy,zz, color = "black", alpha = 0.1)
    # xz plane
    xx, zz = np.meshgrid(np.arange(x_range[0],x_range[1]+x_step, x_step), np.arange(z_range[0],z_range[1]+z_step,z_step))
    yy = np.ones((len(np.arange(z_range[0],z_range[1]+z_step, z_step)), len(np.arange(x_range[0],x_range[1]+x_step, x_step))))*xz_plane
    ax.plot_surface(xx,yy,zz, color = "silver", alpha = 0.1)
    ax.plot_wireframe(xx,yy,zz, color = "black", alpha = 0.1)
        
    # set axis, limit.
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(yz_plane, x_range[1])
    ax.set_ylim(y_range[0],xz_plane)
    ax.set_zlim(xy_plane,z_range[1])
    plt.savefig('Graphs/Outcome_plot_'+ name + '.svg', format = "svg", transparent=True)

def uncert_bar (uncernlst, ylim = 160000):
   
    tier0 = []
    tier1 = []
    tier2 = []
    tier3 = []
    tier4 = []
    tier5 = []
    tier6 = []
    tier7 = []
    for i in uncernlst.ravel():
        if i>=0.7:
            tier7.append(i)
        elif i>=0.6:
            tier6.append(i)
        elif i>=0.5:
            tier5.append(i)
        elif i>=0.4:
            tier4.append(i)
        elif i>=0.3:
            tier3.append(i)
        elif i>=0.2:
            tier2.append(i)
        elif i>=0.1:
            tier1.append(i)
        else:
            tier0.append(i)

    groupnumber = [len(tier0), len(tier1),len(tier2),len(tier3),len(tier4),len(tier5),len(tier6),len(tier7)]
    fig, ax = plt.subplots()
    plt.bar(list(range(8)),groupnumber)
    plt.ylim(0, ylim)
    plt.xlabel('uncertainty')
    plt.ylabel('counts')
    plt.xticks(list(range(8)), ('<0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', "0.5-0.6", "0.6-0.7", ">=0.7"))
    plt.show()
    print("number of points with >0.5 uncertainty (_2ndAL): ", len(tier5)+len(tier6)+len(tier7))

    return groupnumber