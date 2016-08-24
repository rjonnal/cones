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
import geometry


class Cone(geometry.Point):

    def __init__(self,x,y,label):
        self.x = x
        self.y = y
        self.label = label
        
    def __str__(self):
        return '%0.2f,%0.2f,%s'%(self.x,self.y,self.label.lower())

    def __repr__(self):
        return '%0.2f,%0.2f,%s'%(self.x,self.y,self.label.lower())


class Centroider:

    def __init__(self,sy,sx):
        self.xx,self.yy = np.meshgrid(np.arange(sx),np.arange(sy))

    def get(self,im):
        denom = np.sum(im)
        return np.sum(im*self.xx)/denom,np.sum(im*self.yy)/denom


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
    
def bmpify(im):
    return np.clip(np.round(im),0,255).astype(np.uint8)

def cstretch(im):
    return (im - np.min(im))/(np.max(im)-np.min(im))

def high_contrast(r,g):
    out = np.zeros((r.shape[0],r.shape[1],3)).astype(np.uint8)
    out[:,:,0] = bmpify(cstretch(r)*255)
    out[:,:,1] = bmpify(cstretch(g)*255)
    return out

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

def find_cones(im0,cfn,sigma,neighborhood_size):

    im = im0.astype(np.float).mean(axis=2)
    cone_dict = {}
    sy,sx = im.shape
    sx_inches = 7.5
    sy_inches = float(sy)/float(sx)*7.5
    figsize = (sx_inches,sy_inches)
    r = im0[:,:,0]
    g = im0[:,:,1]
    drawing = []
    
    if os.path.exists(cfn):
        temp = np.loadtxt(cfn)
        td = {}
        td[0] = 'S'
        td[1] = 'LM'
        for item in temp:
            cone_dict[(item[0],item[1])] = Cone(item[0],item[1],td[item[2]])
        coords = temp

    else:
        kernel_size = 11
        xx,yy = np.meshgrid(np.arange(kernel_size),np.arange(kernel_size))
        xx = xx.astype(np.float) - (kernel_size-1)/2
        yy = yy.astype(np.float) - (kernel_size-1)/2
        rad = xx**2+yy**2

        g = np.exp(-rad/(2*sigma**2))
        gsum = np.sum(g)
        data = signal.convolve2d(im,g,'same')/gsum


        data_max = filters.maximum_filter(data, neighborhood_size)

        npan = 3
        plt.subplot(1,npan,1)
        plt.cla()
        plt.imshow(im,cmap='gray')
        plt.subplot(1,npan,2)
        plt.cla()
        plt.imshow(data,cmap='gray')
        plt.subplot(1,npan,3)
        plt.cla()
        plt.imshow(data_max)
        plt.show()
        plt.pause(1)

        maxima = (data == data_max)

        labeled, num_objects = ndimage.label(maxima)

        slices = ndimage.find_objects(labeled)
        x, y = [], []

        for dy,dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            x.append(x_center)
            y_center = (dy.start + dy.stop - 1)/2    
            y.append(y_center)

        coords = []
        s_coords = []

        s = im0[:,:,1]
        lm = im0[:,:,0]

        for xi,yi in zip(x,y):
            smean = np.mean(s[yi-1:yi+2,xi-1:xi+2])
            lmmean = np.mean(lm[yi-1:yi+2,xi-1:xi+2])
            if smean>lmmean:
                coords.append([xi,yi,0])
                cone_dict[(xi,yi)] = Cone(xi,yi,'S')
                s_coords.append([xi,yi])
            else:
                coords.append([xi,yi,1])
                cone_dict[(xi,yi)] = Cone(xi,yi,'LM')


        fid = open(cfn,'w')
        for cone in cone_dict.values():
            fid.write('%s\n'%cone)
        fid.close()
        
    #     sys.exit()

    #     plt.figure()
    #     plt.imshow(im,cmap='gray')
    #     plt.figure()
    #     plt.imshow(im,cmap='gray')
    #     plt.autoscale(False)
        
    #     for xi,yi in zip(x,y):
    #         plt.plot(xi,yi,'r.')
    #     plt.show()
    #     sys.exit()

    #     coords = np.array(coords)
    #     np.savetxt(cfn,coords)
    #     coords = coords

    # lm_coords = []
    # s_coords = []
    # for cone in cone_dict.values():
    #     if cone.label=='S':
    #         s_coords.append([cone.x,cone.y])
    #     else:
    #         lm_coords.append([cone.x,cone.y])
    #     cone_dict[(cone.x,cone.y)]=cone

    # lm_coords = np.array(lm_coords)
    # s_coords = np.array(s_coords)
