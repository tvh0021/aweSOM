import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable

import h5py as h5
import sys, os
import matplotlib.ticker as ticker

np.random.seed(0)


class Box:

    #location of (0,0,0) corner
    x0 = 0.0
    y0 = 0.0
    z0 = 0.0

    # box size
    dx = 1.0
    dy = 1.0*dx
    dz = 1.0*dy

    #default plotting style
    #fmt={'color':'k','linestyle':'solid',}


    #ctor
    def __init__(self, ax_in):
        self.ax = ax_in

    def set_data(self, data):
        self.data = data

        self.vmin = np.min( data )
        self.vmax = np.max( data )
        print("vmin = {}".format(self.vmin))
        print("vmax = {}".format(self.vmax))


    # initialize box corners
    def make_corners(self):
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0

        dx = self.dx
        dy = self.dy
        dz = self.dz
        
        #xyz corners
        self.corners = np.array(list(product([x0, x0+dx], [y0, y0+dy], [z0, z0+dz]))) 

        #print("corners are:")
        #print(self.corners)

    # filter given points away from corner array to get visibility conditions
    def filter_points(self, pps):
        corners = []

        #print("filttering...")
        self.make_corners() #update corners; just in case
        for pp in self.corners:
            #print(" drawing ", pp)

            flag=True #there is no such a point
            for ps in pps:
                if (pp == ps).all():
                    flag = False

            if flag:
                #print("     accepted")
                corners.append( pp )
        return corners


    def make_panel(self, corners):
        panel = []
        
        #draw cube
        #print("combinatorics for:", corners)
        for s, e in combinations( corners, 2):
            if np.sum(np.abs(s-e)) == self.dx or \
               np.sum(np.abs(s-e)) == self.dy or \
               np.sum(np.abs(s-e)) == self.dz:
                #print("XX    ", s,e )
                panel.append( (s,e) )

        return panel


    # initialize box outlines
    def draw_outline(self,
            fmt={'color':'k','linestyle':'dashed', 'lw':0.4, 'zorder':20}
            ):
        self.make_corners()

        #outlines = self.make_panel(self.corners)
        #for (p0, p1) in outlines:
        #    #print("connecting ({},{},{}) to ({},{},{})".format(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]))
        #    lines, = self.ax.plot( [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]] )
        #    lines.set(**self.fmt)
        #print("-------------filttering...")

        corners_f = self.filter_points( [ np.array([
            self.x0+self.dx, 
            self.y0+self.dy, 
            self.z0]) ] )
        outlines2 = self.make_panel( corners_f )

        #print(outlines2)
        for (p0, p1) in outlines2:
            #print("connecting ({},{},{}) to ({},{},{})".format(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]))
            lines, = self.ax.plot( [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]] )
            lines.set(**fmt)


    def _check_for_point(self, arr):
        self.make_corners()
        farr = []
        for pp in self.corners:
            #print("drawing", pp, pp[2])
            for i in [0,1,2]:
                if not(arr[i] == None) and not(pp[i] == arr[i]):
                    #print("appending")
                    farr.append(pp)
        return farr

    def _add_offset(self, cors, offs):
        cors2 = []
        for i in range(len(cors)):
            cors2.append( cors[i] + offs )
        return cors2


    # draw also backside and bottom panels using exploded view
    def draw_exploded_panels_outline(self, 
            side, 
            off = dx,
            fmt={'color':'k','linestyle':'dashed',},
            ):

        if side == "bottom":
            farr = self._check_for_point([None, None, self.z0])
            cors = self.filter_points( farr )
            cors2 = self._add_offset(cors, np.array([0,0,-off]) )
            outlines = self.make_panel( cors2 )

        elif side == "left":
            farr = self._check_for_point([self.x0, None, None])
            cors = self.filter_points( farr )
            cors2 = self._add_offset(cors, np.array([-off,0,0]) )
            outlines = self.make_panel( cors2 )

        elif side == "right":
            farr = self._check_for_point([self.x0+self.dx,None, None])
            cors = self.filter_points( farr )
            cors2 = self._add_offset(cors, np.array([+off,0,0]) )
            outlines = self.make_panel( cors2 )

        elif side == "back":
            farr = self._check_for_point([None, self.y0+self.dy,None])
            cors = self.filter_points( farr )
            cors2 = self._add_offset(cors, np.array([0,+off,0]) )
            outlines = self.make_panel( cors2 )


        #print(outlines)
        for (p0, p1) in outlines:
            #print("connecting ({},{},{}) to ({},{},{})".format(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]))
            lines, = self.ax.plot( [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]] )
            lines.set(**fmt)


    def draw_exploded_slice(
            self, 
            side, 
            loc,
            off, 
            cmap='viridis',
            fmt={'color':'k','linestyle':'dotted', 'linewidth':0.5, 'zorder':20},
            fmtC={'color':'k','linestyle':'dotted', 'linewidth':0.3,'zorder':20},
            ):

        if side == "left-front":
            farr = self._check_for_point([self.x0, None, None])
            cors = self.filter_points( farr )
            cors2 = self._add_offset(cors, np.array([loc,-off,0]) )
            outlines = self.make_panel( cors2 )

            #find point in same dim as offset is inserted that is closest to the box
            pp1 = (self.x0 + loc, self.y0, self.z0        )
            pp2 = (self.x0 + loc, self.y0, self.z0+self.dz)
            lines, = self.ax.plot( [pp1[0], pp1[0]], [pp1[1], pp1[1]-off], [pp1[2], pp1[2]], **fmtC)
            lines, = self.ax.plot( [pp2[0], pp2[0]], [pp2[1], pp2[1]-off], [pp2[2], pp2[2]], **fmtC)

            nx,ny,nz = np.shape(self.data)
            iloc = np.int( (loc/self.dx)*nx )
            #print(loc, iloc)
            data_slice = self.data[iloc,:,:]

            ny, nz = np.shape(data_slice)
            Y, Z = np.meshgrid( 
                    np.linspace(self.y0-off, self.y0 + self.dy-off, ny), 
                    np.linspace(self.z0, self.z0 + self.dz, nz))

            x1 = self.x0 + loc
            X = x1*np.ones(Y.shape)


        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)

        #print(outlines)
        for (p0, p1) in outlines:
            #print("connecting ({},{},{}) to ({},{},{})".format(p0[0],p0[1],p0[2], p1[0],p1[1],p1[2]))
            lines, = self.ax.plot( [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]] )
            lines.set(**fmt)





    def draw_top(self, cmap=plt.cm.viridis, data_slice=None):
        farr = self._check_for_point([None, None, self.z0 + self.dz])
        cors = self.filter_points( farr )

        try:
            if data_slice == None:
                data_slice = self.data[:,:,-1]
        except:
            pass

        nx, ny = np.shape(data_slice)
        X, Y = np.meshgrid( 
                np.linspace(self.x0, self.x0 + self.dx, nx), 
                np.linspace(self.y0, self.y0 + self.dy, ny))
        z1 = self.z0 + self.dz
        Z = z1*np.ones(X.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)

    def draw_left(self, cmap=plt.cm.viridis, data_slice=None):
        farr = self._check_for_point([self.x0, None, None])
        cors = self.filter_points( farr )

        try:
            if data_slice == None:
                data_slice = self.data[0,:,:]
        except:
            pass

        ny, nz = np.shape(data_slice)
        Y, Z = np.meshgrid( 
                np.linspace(self.y0, self.y0 + self.dy, ny), 
                np.linspace(self.z0, self.z0 + self.dz, nz))
        x1 = self.y0 
        X = x1*np.ones(Y.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)

    def draw_front(self, cmap=plt.cm.viridis, data_slice=None):
        farr = self._check_for_point([self.x0, None, None])
        cors = self.filter_points( farr )

        try:
            if data_slice == None:
                data_slice = self.data[:,0,:]
        except:
            pass

        nx, nz = np.shape(data_slice)
        X, Z = np.meshgrid( 
                np.linspace(self.x0, self.x0 + self.dx, nx), 
                np.linspace(self.z0, self.z0 + self.dz, nz))
        y1 = self.y0 
        Y = y1*np.ones(X.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)

    def draw_right(self, cmap=plt.cm.viridis, data_slice=None):
        farr = self._check_for_point([self.x0 + self.dx, None, None])
        cors = self.filter_points( farr )

        try:
            if data_slice == None:
                data_slice = self.data[-1,:,:] 
        except:
            pass

        ny, nx = np.shape(data_slice)
        Y, Z = np.meshgrid( 
                np.linspace(self.y0, self.y0 + self.dy, nx), 
                np.linspace(self.z0, self.z0 + self.dz, ny))
        x1 = self.x0 + self.dx
        X = x1*np.ones(Z.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)


    def draw_exploded_bottom(self, off=dx, cmap=plt.cm.inferno, data_slice=None):
        farr = self._check_for_point([None, None, self.z0])
        cors = self.filter_points( farr )

        try:
            if data_slice == None:
                data_slice = self.data[:,:,0] 
        except:
            pass

        nx, ny = np.shape(data_slice)
        X, Y = np.meshgrid( 
                np.linspace(self.x0, self.x0 + self.dx, nx), 
                np.linspace(self.y0, self.y0 + self.dy, ny))
        z1 = self.z0 - off
        Z = z1*np.ones(X.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)


    def draw_exploded_back(self, off=dx, cmap=plt.cm.inferno, data_slice=None):
        farr = self._check_for_point([self.x0+self.dx, self.y0+self.y0, None])
        cors = self.filter_points( farr )

        try:
            if data_slice == None:
                data_slice = self.data[:,-1,:] 
        except:
            pass

        nx, nz = np.shape(data_slice)
        X, Z = np.meshgrid( 
                np.linspace(self.x0, self.x0 + self.dx, nx), 
                np.linspace(self.z0, self.z0 + self.dz, nz))
        y1 = self.y0 + self.dy + off
        Y = y1*np.ones(X.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)

    def draw_exploded_left(self, off=dx, cmap=plt.cm.inferno, data_slice=None):
        farr = self._check_for_point([self.x0, None, None])
        cors = self.filter_points( farr )

        try:
            if data_slice == None:
                data_slice = self.data[0,:,:]
        except:
            pass

        ny, nz = np.shape(data_slice)
        Y, Z = np.meshgrid( 
                np.linspace(self.y0, self.y0 + self.dy, ny), 
                np.linspace(self.z0, self.z0 + self.dz, nz))

        x1 = self.x0 - off
        X = x1*np.ones(Y.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)


    def draw_exploded_right(self, off=dx, cmap=plt.cm.inferno, data_slice=None):
        farr = self._check_for_point([self.x0+self.dx, None, None])
        cors = self.filter_points( farr )

        try:
            if data_slice == None:
                data_slice = self.data[-1,:,:] 
        except:
            pass

        ny, nz = np.shape(data_slice)
        Y, Z = np.meshgrid( 
                np.linspace(self.y0, self.y0 + self.dy, ny), 
                np.linspace(self.z0, self.z0 + self.dz, nz))

        x1 = self.x0 + self.dx + off
        X = x1*np.ones(Y.shape)

        self.draw_surface(X,Y,Z,data_slice, cmap=cmap)


    # mimick volume rendering with multiple opaque slices
    def draw_volume(self):

        nx, ny, nz = np.shape(self.data)
        X, Y = np.meshgrid( 
                np.linspace(self.x0, self.x0 + self.dx, nx), 
                np.linspace(self.y0, self.y0 + self.dy, ny))

        for k in range(ny):
            data_slice = self.data[:,:,k]

            z1 = self.z0 + self.dz*float(k)/float(nz)
            Z = z1*np.ones(X.shape)

            self.draw_surface(X,Y,Z,data_slice, alpha=0.3)




    def draw_surface(self, 
            X, Y, Z, 
            data, 
            cmap = plt.cm.viridis,
            alpha=1.0
            ):

        #print("draw_surface {} {}".format(self.vmin, self.vmax))
        norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        self.ax.plot_surface(
                X,Y,Z,
                rstride=1,
                cstride=1,
                facecolors=cmap( norm( data.T ) ),
                shade=False,
                alpha=alpha,
                antialiased=True,
                )


    def set_ticklabels(
            self,
            #n_ticks,
            L,
            tick_locations,
            tick_boolean,
            tick_labels,
            label='',
            along='x',
            position = 'top',
            direction = 'out',
            offs = 0.10,
            offs_tick = 0.05,
            offs_label = 0.16,
            rotation=0.0
            ):

        n_ticks = len(tick_locations)


        #if along == 'x' and position == 'top':
        #    p0 = (self.x0         ,  self.y0 + self.dy, self.z0 + self.dz )
        #    p1 = (self.x0 + self.dx, self.y0 + self.dy, self.z0 + self.dz )

        if along == 'x' and position == 'top':
            p0 = (self.x0         ,  self.y0, self.z0 + self.dz )
            p1 = (self.x0 + self.dx, self.y0, self.z0 + self.dz )

        if along == 'x' and position == 'top-front':
            p0 = (self.x0         ,  self.y0, self.z0 + self.dz )
            p1 = (self.x0 + self.dx, self.y0, self.z0 + self.dz )

        if along == 'x' and position == 'bottom':
            p0 = (self.x0         ,  self.y0, self.z0)
            p1 = (self.x0 + self.dx, self.y0, self.z0)

        if along == 'x' and position == 'top-behind':
            p0 = (self.x0         ,  self.y0 + self.dy, self.z0 + self.dz )
            p1 = (self.x0 + self.dx, self.y0 + self.dy, self.z0 + self.dz )

        if along == 'y' and position == 'top':
            p0 = (self.x0, self.y0          , self.z0 + self.dz )
            p1 = (self.x0, self.y0 + self.dy, self.z0 + self.dz )

        if along == 'y' and position == 'top-upper':
            p0 = (self.x0+self.dx, self.y0          , self.z0 + self.dz )
            p1 = (self.x0+self.dx, self.y0 + self.dy, self.z0 + self.dz )

        if along == 'y' and position == 'top-front':
            p0 = (self.x0, self.y0          , self.z0 + self.dz )
            p1 = (self.x0, self.y0 + self.dy, self.z0 + self.dz )

        if along == 'y' and position == 'bottom':
            p0 = (self.x0, self.y0          , self.z0)
            p1 = (self.x0, self.y0 + self.dy, self.z0)

        if along == 'z' and position == 'top':
            p0 = (self.x0, self.y0 + self.dy, self.z0           )
            p1 = (self.x0, self.y0 + self.dy, self.z0 + self.dz )

        if along == 'z' and position == 'top-front':
            p0 = (self.x0 + self.dx, self.y0, self.z0           )
            p1 = (self.x0 + self.dx, self.y0, self.z0 + self.dz )

        if along == 'z' and position == 'bottom':
            p0 = (self.x0, self.y0, self.z0           )
            p1 = (self.x0, self.y0, self.z0 + self.dz )

        if along == 'z' and position == 'bottom-front':
            p0 = (self.x0, self.y0, self.z0           )
            p1 = (self.x0, self.y0, self.z0 + self.dz )

        # start and ending points
        x0,y0,z0 = p0
        x1,y1,z1 = p1

        xs = np.linspace(x0,x1,n_ticks)
        ys = np.linspace(y0,y1,n_ticks)
        zs = np.linspace(z0,z1,n_ticks)

        if direction == 'in':
            offs *= -1
            offs_tick *= -1

        # z directions are flipped to point in
        if along == 'z' and position == 'top':
            offs *= -1
            #offs_tick *= -1

        xsa = np.linspace(x0,x1,n_ticks)
        ysa = np.linspace(y0,y1,n_ticks)
        zsa = np.linspace(z0,z1,n_ticks)

        xst = np.linspace(x0,x1,n_ticks)
        yst = np.linspace(y0,y1,n_ticks)
        zst = np.linspace(z0,z1,n_ticks)

        # along x so we push the ticks towards y
        if along == 'x':
            xs  = tick_locations*self.dx/L
            xsa = tick_locations*self.dx/L
            xst = tick_locations*self.dx/L

            if position == 'top':
                ysa -= offs
            if position == 'bottom':
                zsa -= offs
            if position == 'top-behind':
                ysa += offs
            if position == 'top-front':
                zsa += offs

            yst += offs_tick

        # along y so we push the ticks towards x
        if along == 'y':
            ys  = tick_locations*self.dy/L
            ysa = tick_locations*self.dy/L
            yst = tick_locations*self.dy/L

            if position =='top':
                xsa -= offs
            if position =='top-upper':
                xsa += offs
            if position =='bottom':
                zsa -= offs
            if position =='top-front':
                zsa += offs

            xst += offs_tick

        # along z so we push the ticks towards x
        if along == 'z':
            zs  = tick_locations*self.dz/L
            zsa = tick_locations*self.dz/L
            zst = tick_locations*self.dz/L

            if position =='top':
                ysa -= offs
            if position =='top-front':
                xsa -= offs
            if position =='bottom':
                xsa -= offs
            if position =='bottom-front':
                ysa -= offs

            xst += offs_tick



        for i in range(n_ticks):
            self.ax.plot(
                    [xs[i], xsa[i]],
                    [ys[i], ysa[i]],
                    [zs[i], zsa[i]],
                    color='k',
                    linestyle='solid',
                    linewidth=0.5,
                    zorder=30,
                    )

            if along == 'x':
                val = xs[i]*L
            if along == 'y':
                val = ys[i]*L
            if along == 'z':
                val = zs[i]*L
            val = round(val,1)
            #print("setting tick to:", val)
    
            #print("testing tick label to:", xst[i], yst[i], zst[i])

            #if val in tick_labels:
            if len(tick_boolean) > 0 and tick_boolean[i]:

                #stick = "{:.1f}".format(ticklabels[i])
                #stick = "{:.1f}".format(val)

                #stick = "{:3d}".format(int(val))
                #print('i', i)
                #print(tick_labels[i])
                stick = "{:3d}".format(tick_labels[i])
                #print("        tick label to:", xst[i], yst[i], zst[i], ' str:', stick)

                self.ax.text(
                        xst[i], yst[i], zst[i],
                        stick,
                        va='center',
                        ha='center',
                        fontsize=5,
                        zorder=30,
                        )
                        

        # label goes to half way of points
        xhf = x0 + (x1 + x0)/2
        yhf = y0 + (y1 + y0)/2
        zhf = z0 + (z1 + z0)/2
        #print("label loc:", xhf, yhf, zhf)

        if along =='x' and position == 'bottom':
            yhf -= offs_label
            zhf = 0.0
        if along =='y' and position == 'bottom':
            xhf -= offs_label
            #yhf = 0.3
            zhf = 0.0
        if along =='z' and position == 'top':
            zhf = 0.5
            yhf = 1.0
            xhf -= offs_label

        self.ax.text(
                xhf, yhf, zhf,
                label,
                va='center',
                ha='center',
                fontsize=5,
                zorder=30,
                rotation=rotation,
                )


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)



#for data
def f(x, y, z):
    #u = np.cos(x)*np.sin(y)
    #v = np.sin(x)*np.cos(y)
    #w = np.cos(x)*np.cos(y)
    kx = 1.0
    ky = 5.0
    kz = 10.0
    return np.cos(kx*x + 0.1) + np.cos(ky*y + 0.2) + np.cos(kz*z + 0.3)

#def f(x, y, z):
#    return np.exp(-0.5*(x - 1.0)**2.0) * np.exp(-0.10*(y - 1.0)**2.0) * np.exp(-0.01*(z - 1.0)**2.0)


#def f(x, y, z):
#    sig1 = 1.0
#    sig2 = 2.0
#    sig3 = 3.0
#    norm1 = 1.0/np.sqrt(2.0*np.pi*sig1)
#    norm2 = 1.0/np.sqrt(2.0*np.pi*sig2)
#    norm3 = 1.0/np.sqrt(2.0*np.pi*sig3)
#    return norm1*norm2*norm3*np.exp(-0.5*((x - 1.0)/sig1)**2.0) * np.exp(-0.5*((y - 1.0)/sig2)**2.0) * np.exp(-0.5*((z - 1.0)/sig3)**2.0) 



##################################################
# plot
if __name__ == "__main__":

    
    fig = plt.figure(figsize=(3.54, 3.0)) #single column fig
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    
    plt.rc('font', family='serif', size=7)
    plt.rc('xtick')
    plt.rc('ytick')
    
    axs = []
    axs.append( fig.add_subplot(111, projection='3d') )
    
    #create data
    Nx = 50
    x = np.linspace(0.0, 1.0, Nx)
    y = np.linspace(0.0, 1.0, Nx)
    z = np.linspace(0.0, 1.0, Nx)
    X, Y, Z = np.meshgrid(x, y, z)
    data = f(X,Y,Z)


    #random data; low-k forcing
    #data = np.zeros((Nx, Nx, np.int(Nx/2)+1 ),dtype=np.complex64)
    #for i in range(1,10):
    #    for j in range(1,10):
    #        for k in range(1,5):
    #            if np.sqrt(i*i + j*j + k*k) > 6:
    #                continue
    #            norm = 1.0/(i + j + k + 1)**2.0
    #            val = np.random.randn() + 1j*np.random.randn()
    #            val /= val*val
    #            data[i,j,k] = norm*val
    #data = np.fft.irfftn(data)
    #print(np.shape(data))

    ################################################## 
    # draw box
    

    box = Box(axs[0])
    box.set_data(data)

    #surface rendering
    box.draw_top()
    box.draw_left()
    box.draw_right()

    #volume rendering
    #box.draw_volume()
    box.draw_outline()


    #back exploded panels
    if True:
        off = 0.7
        box.draw_exploded_panels_outline("bottom", off=off)
        box.draw_exploded_panels_outline("left",   off=off)
        box.draw_exploded_panels_outline("right",  off=off)
        
        box.draw_exploded_bottom(off=off)
        box.draw_exploded_left(  off=off)
        box.draw_exploded_right( off=off)

    if True:
        #front exploded panels
        off = -1.95 #this puts them in front of the box
        cmap = plt.cm.RdBu
        box.draw_exploded_panels_outline("bottom", off=off)
        box.draw_exploded_panels_outline("left",   off=off)
        box.draw_exploded_panels_outline("right",  off=off)
        
        box.draw_exploded_bottom(off=off, cmap=cmap)
        box.draw_exploded_left(  off=off, cmap=cmap)
        box.draw_exploded_right( off=off, cmap=cmap)


    axs[0].set_axis_off()
    axs[0].view_init(35.0, 45.0)


    if False:
        #colorbars
        m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        m.set_array([0.0, 1.0])
        m.set_clim( vmin=0.0, vmax=1.0 )
        cbaxes = fig.add_axes([0.2, 0.91, 0.6, 0.02]) #[left, bottom, width, height],
        cb = plt.colorbar(m, cax = cbaxes, orientation="horizontal", ticklocation="top")  
        fig.text(0.15, 0.91,  r'$n_{\pm}$')

        m = plt.cm.ScalarMappable(cmap=plt.cm.inferno)
        m.set_array([0.0, 1.0])
        m.set_clim( vmin=0.0, vmax=1.0 )
        cbaxes = fig.add_axes([0.2, 0.09, 0.6, 0.02]) #[left, bottom, width, height],
        cb = plt.colorbar(m, cax = cbaxes, orientation="horizontal", ticklocation="top")  
        fig.text(0.15, 0.10,  r'$n_{\nu}$')

        m = plt.cm.ScalarMappable(cmap=plt.cm.RdBu)
        m.set_array([-1.0, 1.0])
        m.set_clim( vmin=-1.0, vmax=1.0 )
        cbaxes = fig.add_axes([0.2, 0.06, 0.6, 0.02]) #[left, bottom, width, height],
        cb = plt.colorbar(m, cax = cbaxes, orientation="horizontal", ticklocation="bottom")  
        #cb.set_label(r'$J$', rotation=0)
        fig.text(0.15, 0.05,  r'$J$')



    axisEqual3D(axs[0])
    fname = 'box'
    plt.subplots_adjust(left=-0.1, bottom=-0.1, right=1.1, top=1.1)
    plt.savefig(fname+'.pdf')
    plt.savefig(fname+'.png')
    
