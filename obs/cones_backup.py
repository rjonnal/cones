""" A set of tools useful for cone analysis.
"""

from scipy.misc import imread,factorial
from glob import glob
import sys,os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import morphology
from scipy.ndimage import filters
from scipy import ndimage
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
import csv

class DataSet:

    def __init__(self,tif_fn):
        self.tif_fn = tif_fn
        head,tail = os.path.split(tif_fn)
        head,pretail = os.path.split(head)
        self.tag = '%s_%s'%(pretail,tail)
        self.im0 = imread(tif_fn)
        self.im = self.im0.astype(np.float).mean(axis=2)
        self.sy,self.sx = self.im.shape
        self.coordinate_fn = self.tif_fn.replace('images_original','cone_coordinates').replace('.tif','.txt')
        self.filtered_fn = self.coordinate_fn.replace('.txt','.npy')
        self.xx,self.yy = np.meshgrid(np.arange(self.sx),np.arange(self.sy))
        print 'input file %s'%self.tif_fn
        print 'coordinate file %s'%self.coordinate_fn
        if self.has_coordinates():
            self.cc = ConeCoordinates(self.coordinate_fn,self.filtered_fn)

    def has_coordinates(self):
        return os.path.exists(self.coordinate_fn) and os.path.exists(self.filtered_fn)
        
    def get_label(self,x,y,rad=5):
        d = np.sqrt((self.xx-x)**2+(self.yy-y)**2)
        d[np.where(d>rad)] = 0
        d[np.where(d)] = 1
        rtest = np.sum(self.im0[:,:,0]*d)
        gtest = np.sum(self.im0[:,:,1]*d)
        if rtest>gtest:
            return 'l/m'
        else:
            return 's'
        
    def imshow(self,dpi=100):
        plt.figure(figsize=(self.sx/dpi,self.sy/dpi))
        plt.axes([0,0,1,1])
        plt.imshow(self.im0)
        plt.xticks([])
        plt.yticks([])

    def show_channels(self,factor=100,bs=False):
        if bs:
            to_show = background_subtract(self.im0)
        else:
            to_show = self.im0

        fsy,fsx = self.sy/factor,self.sx/factor*3
        plt.figure(figsize=(fsx,fsy))
        plt.axes([0,0,.33,1.0])
        plt.imshow(to_show[:,:,0])
        plt.axes([0.33,0,.33,1.0])
        plt.imshow(to_show[:,:,1])
        plt.axes([0.66,0,.33,1.0])
        plt.imshow(to_show[:,:,2])

class Centroider:

    def __init__(self,sy,sx):
        self.xx,self.yy = np.meshgrid(np.arange(sx),np.arange(sy))

    def get(self,im):
        denom = np.sum(im)
        return np.sum(im*self.xx)/denom,np.sum(im*self.yy)/denom

class ConeCoordinates:

    def __init__(self,cfn=None,ffn=None):
        self.x = []
        self.y = []
        self.labels = []
        if cfn is not None and ffn is not None:
            self.filtered = np.load(ffn)
            with open(cfn,'rb') as csvfile:
                reader = csv.reader(csvfile,delimiter=',')
                for row in reader:
                    self.x.append(float(row[0]))
                    self.y.append(float(row[1]))
                    self.labels.append(row[2])
        
                
            
    
    def add(self,x,y,label):
        self.x = self.x+x
        self.y = self.y+y
        self.labels = self.labels+[label]*len(x)

    def set_filtered(self,im):
        self.filtered = im
        
    def save(self,fn,delimiter=','):
        print 'saving to %s'%fn
        fid = open(fn,'w')
        for x,y,label in zip(self.x,self.y,self.labels):
            outstr = '%0.2f%s%0.2f%s%s%s\n'%(x,delimiter,y,delimiter,label,delimiter)
            fid.write(outstr)
            print outstr[:-1]
        fid.close()
        ffn = fn.replace('.txt','.npy')
        np.save(ffn,self.filtered)
        
    def plot(self,axes=None):

        if axes is None:
            try:
                axes = plt.gca()
            except Exception as e:
                axes = plt.axes()
                
        plt.autoscale(False)

        for x,y,label in zip(self.x,self.y,self.labels):
            if label=='l/m':
                ph = axes.plot(x,y,'ro')
                ph[0].set_markeredgecolor('w')
                ph[0].set_markerfacecolor('none')
                ph[0].set_markersize(4)
            if label=='s':
                ph = axes.plot(x,y,'gs')
                ph[0].set_markeredgecolor('w')
                ph[0].set_markerfacecolor('none')
                ph[0].set_markersize(4)

    def find(self,x,y,tol=5.0):
        d = np.sqrt((np.array(self.x)-x)**2+(np.array(self.y)-y)**2)
        if np.min(d)<tol:
            return np.argmin(d)
        else:
            return -1

    def edit(self,x,y,label='?'):
        idx = self.find(x,y)
        print idx
        if idx>-1:
            self.x = self.x[:idx] + self.x[idx+1:]
            self.y = self.y[:idx] + self.y[idx+1:]
            self.labels = self.labels[:idx] + self.labels[idx+1:]
        else:
            self.x.append(x)
            self.y.append(y)
            self.labels.append(label)


def strel(kind='disk',diameter=15):
    if kind=='disk':
        xx,yy = np.meshgrid(np.arange(diameter),np.arange(diameter))
        xx = xx - float(diameter-1)/2.0
        yy = yy - float(diameter-1)/2.0
        d = np.sqrt(xx**2+yy**2)
        out = np.zeros(xx.shape)
        out[np.where(d<=diameter/2.0)] = 1.0
        return out

def background_subtract(im,strel=strel()):
    if len(im.shape)==2:
        bg = morphology.grey_opening(im,structure=strel)
    elif len(im.shape)==3:
        bg = np.zeros(im.shape)
        for k in range(3):
            bg[:,:,k] = morphology.grey_opening(im[:,:,k],structure=strel)
    return im-bg
    
def gaussian_blur(im,sigma=1.0,kernel_size=11):
    xx,yy = np.meshgrid(np.arange(kernel_size),np.arange(kernel_size))
    xx = xx.astype(np.float) - (kernel_size-1)/2
    yy = yy.astype(np.float) - (kernel_size-1)/2
    rad = xx**2+yy**2
    g = np.exp(-rad/(2*sigma**2))
    gsum = np.sum(g)
    if len(im.shape)==2:
        data = signal.convolve2d(im,g,'same')/gsum
    elif len(im.shape)==3:
        data = np.zeros(im.shape)
        for idx in range(im.shape[2]):
            data[:,:,idx] = signal.convolve2d(im[:,:,idx],g,'same')/gsum
    return data
    
def autocorrelation(im):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(im)*np.conj(np.fft.fft2(im))))

def bmpify(im):
    return np.clip(np.round(im),0,255).astype(np.uint8)

def showset2(im_list,factor=150.0,cmap='jet',rx=[],ry=[],gx=[],gy=[],animation_duration=0.0,fn=None):
    factor = float(factor)
    nrows = len(im_list)
    ncols = 1
    for item in im_list:
        if len(item)>ncols:
            ncols = len(item)
    dx = 1.0/ncols
    dy = 1.0/nrows

    try:
        sy,sx = im_list[0][0].shape
    except ValueError as ve:
        sy,sx,nc = im_list[0][0].shape

    fsx = sx * ncols / factor
    fsy = sy * nrows / factor
    plt.figure(figsize=(fsx,fsy))
    x = 0.0
    y = 1.0-dy

    for row_idx,im_set in enumerate(im_list):
        for col_idx,im in enumerate(im_set):
            if type(im)==tuple:
                im = im[0]
                mycmap = im[1]
            else:
                mycmap = cmap
            if len(im.shape)==3:
                im = bmpify(im)

            ax = plt.axes([x,y,dx,dy])
            imh = plt.imshow(im)
            plt.set_cmap(mycmap)
            plt.xticks([])
            plt.yticks([])


            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="2%", pad=0.05)
            #plt.colorbar(imh, cax=cax)

            plt.autoscale(False)

            ph = plt.plot(rx,ry,'ro')
            ph[0].set_markeredgecolor('w')
            ph[0].set_markerfacecolor('y')

            ph = plt.plot(gx,gy,'go')
            ph[0].set_markeredgecolor('w')
            ph[0].set_markerfacecolor('b')

            x = x + dx
        y = y - dy
        x = 0.0
        
    plt.show()
    

def showset(im_list,factor=50.0,cmap='jet',rx=[],ry=[],gx=[],gy=[],animation_duration=0.0,fn=None):

    factor = float(factor)
    nim = float(len(im_list))
    dx = 1.0/nim
    try:
        sy,sx = im_list[0].shape
    except ValueError as ve:
        sy,sx,nc = im_list[0].shape
        
    fsx = sx * nim / factor
    fsy = sy / factor
    plt.figure(figsize=(fsx,fsy))
    x = 0.0
    for idx,im in enumerate(im_list):
        if len(im.shape)==3:
            im = bmpify(im)
        ax = plt.axes([x,0,dx,1])
        imh = plt.imshow(im)

        
        if type(cmap)==list:
            cidx = idx%len(cmap)
            if cmap[cidx] is not None:
                plt.set_cmap(cmap[cidx])
        else:
            plt.set_cmap(cmap)
        plt.xticks([])
        plt.yticks([])


        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="2%", pad=0.05)
        #plt.colorbar(imh, cax=cax)

        plt.autoscale(False)

        ph = plt.plot(rx,ry,'ro')
        ph[0].set_markeredgecolor('w')
        ph[0].set_markerfacecolor('y')

        ph = plt.plot(gx,gy,'go')
        ph[0].set_markeredgecolor('w')
        ph[0].set_markerfacecolor('b')
        
        x = x + dx

    if fn is not None:
        plt.savefig(fn,dpi=100)
        
    if animation_duration:
        plt.pause(animation_duration)
        plt.close()
    else:
        plt.show()

def quarter(im):
    if len(im.shape)==2:
        sy,sx = im.shape
        return im[sy/4:3*sy/4,sx/4:3*sx/4]
    else:
        sy,sx,nc = im.shape
        return im[sy/4:3*sy/4,sx/4:3*sx/4,:]

def cstretch(im):
    return (im - np.min(im))/(np.max(im)-np.min(im))

def high_contrast(r,g):
    out = np.zeros((r.shape[0],r.shape[1],3)).astype(np.uint8)
    out[:,:,0] = bmpify(cstretch(r)*255)
    out[:,:,1] = bmpify(cstretch(g)*255)
    return out

def flatten(im):
    r = im[:,:,0].astype(np.float)
    g = im[:,:,1].astype(np.float)
    return cstretch(r)+cstretch(g)


def get_mask(im,nstd,wthresh_fraction):

    imvec = im.ravel()
    imthresh = np.zeros(im.shape)
    imthresh[:] = im[:]

    thresh = np.mean(imvec)+np.std(imvec)*nstd
    imthresh[np.where(im<thresh)] = 0

    imlabeled,im_num_objects = ndimage.label(imthresh)

    immask = np.zeros(im.shape)

    imweights = []
    ks = []
    for k in range(1,im_num_objects):
        weight = np.sum(im[np.where(imlabeled==k)])
        imweights.append(weight)
        ks.append(k)
        
    imweights = np.array(imweights)
    ks = np.array(ks)

    wthresh = np.mean(imweights)*wthresh_fraction
        
    valid = np.where(imweights>wthresh)[0]

    imweights = imweights[valid]
    ks = ks[valid]
    
    for k in ks:
        immask[np.where(imlabeled==k)] = 1.0
        
    #showset([im,imthresh,immask])
    return immask

def centroid_objects(im,mask):
    cyvec,cxvec = [],[]
    sy,sx = im.shape
    c = Centroider(sy,sx)
    labeled,n_objects = ndimage.label(mask)
    for k in range(1,n_objects):
        tempmask = np.zeros(mask.shape)
        tempmask[np.where(labeled==k)] = 1.0
        tempim = im*tempmask
        cx,cy = c.get(tempim)
        cyvec.append(cy)
        cxvec.append(cx)
    return cxvec,cyvec

def local_normalize(gim,kernel_size):
    maxf = filters.maximum_filter(gim,kernel_size)
    minf = filters.minimum_filter(gim,kernel_size)
    num = gim-minf
    den = maxf-minf
    den[np.where(den==0)]=1.0
    return num/den

def global_normalize(gim):
    num = gim - np.min(gim)
    den = np.max(gim) - np.min(gim)
    if den==0:
        den = 1.0
    return num/den

def find_peaks_1(cim,nstd,wthresh_fraction,sigma,outfn=None):
    #cim = quarter(cim)
    r = cim[:,:,0].astype(np.float)
    g = cim[:,:,1].astype(np.float)
    r = background_subtract(r)
    g = background_subtract(g)

    mask = np.zeros(r.shape)

    edge = 10

    r = gaussian_blur(r,sigma=sigma)
    g = gaussian_blur(g,sigma=sigma)

    gmask = get_mask(g,nstd=nstd,wthresh_fraction=wthresh_fraction)
    rmask = get_mask(r,nstd=nstd,wthresh_fraction=wthresh_fraction)

    rx,ry = centroid_objects(r,rmask)
    gx,gy = centroid_objects(g,gmask)
    
    showset([cim,g,gmask,r,rmask],cmap='gray',rx=rx,ry=ry,gx=gx,gy=gy,fn=outfn,animation_duration=1.0)

    return ConeCoordinates(rx,ry,gx,gy)


def find_peaks_0(cim,sigma,frac,nbins,erosion_diameter):

    r = cim[:,:,0].astype(np.float)
    g = cim[:,:,1].astype(np.float)

    gto = threshold(g,sigma,frac,nbins,erosion_diameter)
    rto = threshold(r,sigma,frac,nbins,erosion_diameter)

    out = high_contrast(rto,gto)
    
    g_mask = np.zeros(gto.shape)
    g_mask[np.where(gto)] = 1.0
    r_mask = np.zeros(rto.shape)
    r_mask[np.where(rto)] = 1.0

    rx,ry = centroid_objects(r,r_mask)
    gx,gy = centroid_objects(g,g_mask)

    cc = ConeCoordinates()
    cc.add(rx,ry,'l/m')
    cc.add(gx,gy,'s')
    cc.set_filtered(out)
    
    return cc,out
    
def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def get_poisson_threshold(g,frac=0.01,nbins=50,p0=4.0):
    normed_counts, bin_edges = np.histogram(g.ravel(),bins=nbins,range=[g.min()-.5,g.max()+.5],normed=True)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    
    # poisson function, parameter lamb is the fit parameter
    parameters, cov_matrix = curve_fit(poisson, bin_centers, normed_counts,p0=[p0])
    x_plot = np.linspace(np.min(bin_centers), np.max(bin_centers), 1000)
    y_plot = poisson(x_plot,*parameters)
    threshold = round(x_plot[np.where(y_plot>frac)[0][-1]])
    return threshold

def threshold(g,sigma,frac,nbins,erosion_diameter):
    g = background_subtract(g,strel())
    g = gaussian_blur(g,sigma)
    g = np.round(g)
    gthreshold = get_poisson_threshold(g,frac=frac,nbins=nbins)
    gt = g.copy()
    gt[np.where(g<gthreshold)] = 0.0
    gto = morphology.grey_erosion(gt,footprint=strel(diameter=erosion_diameter))
    return gto


if __name__=='__main__':
    flist = glob('/home/rjonnal/data/Dropbox/Private/figures/src/opsins_voronoi/images_original/*/*.tif')
    #flist = glob('/home/rjonnal/data/Dropbox/Private/figures/src/opsins_voronoi/images_original/I132G/*.tif')
    #flist = glob('/home/rjonnal/data/Dropbox/Private/figures/src/opsins_voronoi/images_original/M206/*.tif')
    #flist = glob('/home/rjonnal/data/Dropbox/Private/figures/src/opsins_voronoi/images_original/set01/*.tif')

    from cones_gui import *
    for f in flist:
        ds = DataSet(f)
        im = ds.im0
        outfn = os.path.join('/home/rjonnal/Dropbox/Private/figures/src/opsins_voronoi/images_output/',ds.tag+'_cone_identification.png')
        cc = find_peaks_0(im,sigma=2.0,frac=0.005,nbins=50,erosion_diameter=5)
        showset([im],rx=cc.rx,ry=cc.ry,gx=cc.gx,gy=cc.gy,fn=outfn,animation_duration=0.1)

        
