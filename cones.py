""" A set of tools useful for cone analysis.
"""

from scipy.misc import imread,factorial
from glob import glob
import sys,os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import morphology,filters
from scipy.spatial import Voronoi
from scipy import ndimage
from scipy import signal
from scipy.stats import bartlett, fligner, levene
from scipy.stats import f as ftest
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
import csv
import geometry
from cone_tools import threshold,high_contrast,centroid_objects,strel,gaussian_blur,background_subtract

DEFAULT_SIGMA = 2.0
DEFAULT_FRAC = 0.001
DEFAULT_NBINS = 50
DEFAULT_DIAMETER = 5
DEFAULT_MARKER = 'y-'
H0 = 0.00
H1 = .4
HSTEP = 0.002
RIPLEY_MM_RANGE = np.arange(H0,H1,HSTEP)

class Cone(geometry.Point):

    def __init__(self,x,y,label):
        self.x = x
        self.y = y
        self.label = label
        
    def __str__(self):
        return '%0.2f,%0.2f,%s'%(self.x,self.y,self.label)

    def __repr__(self):
        return '%0.2f,%0.2f,%s'%(self.x,self.y,self.label)


class DataSet:

    def __init__(self,tif_fn,ecc_mm,mm_per_pixel):
        self.tif_fn = tif_fn
        head,tail = os.path.split(tif_fn)
        self.dname = head
        head,pretail = os.path.split(head)
        self.tag = '%s_%s'%(pretail,tail)
        self.im0 = imread(tif_fn)
        self.im = self.im0[:,:,:3].astype(np.float).mean(axis=2)
        self.mask = np.zeros(self.im.shape)
        self.mask[np.where(self.im)] = 1.0

        self.mm_per_pixel = mm_per_pixel
        self.area_sqmm = np.sum(self.mask*self.mm_per_pixel**2)

        self.ecc_mm = ecc_mm
        
        self.r = self.im0[:,:,0].astype(np.float)
        self.g = self.im0[:,:,1].astype(np.float)
        self.rbs = background_subtract(self.r,strel())
        self.gbs = background_subtract(self.g,strel())

        self.sy,self.sx = self.im.shape
        self.coordinate_fn = self.tif_fn.replace('images','cone_coordinates').replace('.tif','.txt')
        self.all_distance_fn = self.tif_fn.replace('images','distance_matrices_all').replace('.tif','_distmat_all.csv')
        self.s_distance_fn = self.tif_fn.replace('images','distance_matrices_s').replace('.tif','_distmat_s.csv')
        self.filtered_fn = self.tif_fn.replace('images','images_filtered').replace('.tif','.npy')
        self.marked_fn = self.tif_fn.replace('images','images_marked').replace('.tif','.png')
        self.s_voronoi_fn = self.tif_fn.replace('images','images_marked').replace('.tif','_s-vor.png')

        
        self.xx,self.yy = np.meshgrid(np.arange(self.sx),np.arange(self.sy))
        self.coneset = ConeSet(mask=self.mask)
        try:
            self.coneset.from_file(self.coordinate_fn)
        except Exception as e:
            self.log('Error %s: cannot load %s. Empty cone set.'%(e,self.coordinate_fn))

        self.s_coneset = ConeSet(mask=self.mask)
        try:
            self.s_coneset.from_file(self.coordinate_fn,filt='s')
        except Exception as e:
            self.log('Error %s: cannot load %s. Empty s-cone set.'%(e,self.coordinate_fn))
        
        
        self.do_counts()
        self.filter(DEFAULT_SIGMA,DEFAULT_FRAC,DEFAULT_NBINS,DEFAULT_DIAMETER)
        self.log('input file %s'%self.tif_fn)
        self.log('coordinate file %s'%self.coordinate_fn)
        self.log('filtered file %s'%self.filtered_fn)

    def log(self,msg):
        print msg

    def compute_s_distance_matrix(self,redo=False):
        if not os.path.exists(self.s_distance_fn) or redo:
            self.s_coneset.make_distance_matrix(self.s_distance_fn,self.mm_per_pixel)
        self.s_distance_matrix = np.genfromtxt(self.s_distance_fn,delimiter=",")


    def compute_s_ripley_term(self,cone,h_mm,idx,do_plot=False):
        x = cone.x
        y = cone.y
        xx = self.xx - x
        yy = self.yy - y
        d = np.sqrt(xx**2+yy**2)*self.mm_per_pixel
        mask = np.zeros(d.shape)
        mask[np.where(d<=h_mm)] = 1
        unclipped_area = h_mm**2*np.pi
        clipped_area = np.sum(mask*self.mask)*(self.mm_per_pixel**2)
        
        if h_mm:
            weight = clipped_area/unclipped_area
        else:
            weight = 1.0

        auto_limit = np.sqrt((self.sx*self.mm_per_pixel)**2+(self.sy*self.mm_per_pixel)**2)
        auto_count = h_mm>auto_limit or all(mask.ravel())
        if not auto_count:
            r = list(self.s_distance_matrix[idx,idx+1:])
            c = list(self.s_distance_matrix[:idx,idx])
            ds = np.array(r+c)
            count = len(np.where(ds<=h_mm)[0])
        else:
            count = len(self.s_coneset.cones)
        
        weighted_count = float(count)/weight

        if count and not auto_count and do_plot:
            plt.cla()
            ph = plt.imshow(mask*self.g)
            #ph.set_cmap('gray')
            plt.title('%0.2fmm, %0.2f, %d'%(h_mm,weight,count))
            plt.pause(.001)
        
        return weighted_count

    def compute_s_ripley_matrix(self,redo=False):
        range_str = '%0.3f_%0.3f_%0.3f'%(H0,H1,HSTEP)
        self.s_ripley_fn = self.tif_fn.replace('images','ripley_matrices_s').replace('.tif','_ripley_matrix_s_%s.csv'%range_str)
        self.s_ripley_range_fn = self.tif_fn.replace('images','ripley_matrices_s').replace('.tif','_ripley_matrix_s_mm_%s.csv'%range_str)
        if not os.path.exists(self.s_ripley_fn) or redo:
            mm_range = RIPLEY_MM_RANGE
            result = np.zeros((len(mm_range),len(self.s_coneset.cones)))
            for idx2,cone in enumerate(self.s_coneset.cones):
                for idx1,h_mm in enumerate(mm_range):
                    result[idx1,idx2] = self.compute_s_ripley_term(cone,h_mm,idx2)
                print 'cone %d of %d'%(idx2,len(self.s_coneset))
                print result[:,idx2]
            np.savetxt(self.s_ripley_fn,result,delimiter=',')
            np.savetxt(self.s_ripley_range_fn,mm_range,delimiter=',')

    def get_s_ripley_k_row(self,header=False,show_plot=True):
        range_str = '%0.3f_%0.3f_%0.3f'%(H0,H1,HSTEP)
        self.s_ripley_fn = self.tif_fn.replace('images','ripley_matrices_s').replace('.tif','_ripley_matrix_s_%s.csv'%range_str)
        self.s_ripley_range_fn = self.tif_fn.replace('images','ripley_matrices_s').replace('.tif','_ripley_matrix_s_mm_%s.csv'%range_str)
        mat = np.loadtxt(self.s_ripley_fn,delimiter=',')
        mm_range = np.loadtxt(self.s_ripley_range_fn)
        area = self.mm_per_pixel**2*np.sum(self.mask)

        
        N = float(len(self.s_coneset))
        density = N/area

        circle_areas = mm_range**2*np.pi
        
        K = np.mean(mat,axis=1)/density
        K_csr = mm_range**2*np.pi
        

        def K2L(k):
            return np.sqrt(k/np.pi)-mm_range

        def L2K(l):
            return (l+mm_range)**2*np.pi

        L_csr_05_ci = 1.42 * np.sqrt(area) / (N-1)
        L_csr_01_ci = 1.68 * np.sqrt(area) / (N-1)
        
        L = K2L(K)
        L_csr = K2L(K_csr)

        L_ulim = L_csr + L_csr_05_ci
        L_llim = L_csr - L_csr_05_ci

        K_ulim = L2K(L_ulim)
        K_llim = L2K(L_llim)
        
        x = mm_range


        extremum_idx = np.argmax(np.abs(L))
        extremum = L[extremum_idx]
        lower = L_llim[extremum_idx]
        upper = L_ulim[extremum_idx]
        r_min = x[np.argmin(L)]
        sig = extremum<=lower or extremum>=upper

        L1 = 0.1
        B1 = 0.55
        B2 = 0.1
        B3 = 0.25
        W = 0.45
        H = 0.4
        
        W1 = W-L1
        L2 = L1+W1+0.02
        W2 = .98-L2
        
        
        if header:
            plt.figure(figsize=(11,8.5))
        ax1 = plt.axes([L1,B1,W1,H])
        ax1.clear()
        ax1.plot(x,K,'k')
        ax1.plot(x,K_csr,'b--')
        ax1.fill_between(x,K_llim,K_ulim,where=K_ulim>K_llim,alpha=0.3)
        #ax1.xlabel('$r$ ($mm$)')
        ax1.set_ylabel('$K(r)$ ($mm^2$)')
        ax1.set_ylim((0,0.6))
        
        ax2 = plt.axes([L1,B2,W1,H])
        ax2.clear()
        ax2.plot(x,L,'k')
        ax2.plot(x,L_csr,'b--')
        ax2.fill_between(x,L_llim,L_ulim,where=L_ulim>L_llim,alpha=0.3)
        ax2.set_xlabel('$r$ ($mm$)')
        ax2.set_ylabel('$L(r)$ ($mm$)')
        ax2.set_ylim((-.07,.02))

        ax2.plot(x[extremum_idx],L[extremum_idx],'rs')
        info = '$r_m=%0.3f$,$L_m=%0.3f$'%(x[extremum_idx],L[extremum_idx])
        ax2.text(x[extremum_idx]+.01,L[extremum_idx],info,ha='left',va='center')

        sbb = 5
        
        ax3 = plt.axes([L2,B3,W2,.5])
        ax3.clear()
        ax3.imshow(self.g)
        rpx = np.abs(r_min)/self.mm_per_pixel

        sbh = self.sy/30.
        sbw = 0.1/self.mm_per_pixel
        rbar = plt.Rectangle((sbb,2*sbb+sbh),rpx,sbh,color='w',alpha=0.5)
        ax3.add_artist(rbar)
        plt.text(sbb+rpx+5,2*sbb+sbh+1,'$L_m=%0.2f \mu m$'%(np.abs(L[extremum_idx])*1000.0),color='w',va='top')

        # circ_x = rpx+2
        # circ_y = self.sy-rpx-2
        # circ = plt.Circle((circ_x,circ_y),rpx,color='w',alpha=0.5)
        # ax3.add_artist(circ)
        # plt.text(circ_x+rpx+sbb,circ_y,'$r/2=%0.1f \mu m$'%(np.abs(r_min)/2.0*1000.0),color='w',va='center')
        
        scalebar = plt.Rectangle((sbb,sbb),sbw,sbh,color='w',alpha=0.5)
        ax3.add_artist(scalebar)
        plt.text(2*sbb,sbb-1,'$100 \mu m$',color='k',va='top')

        
        plt.savefig(self.s_ripley_fn.replace('.csv','_plot.png'))

        
        if show_plot:
            plt.pause(.1)



        exp_density = (0.9306/r_min/np.cos(np.pi/6.0))**2
            
        if header:
            return ['file_id','1/max(abs(L)) (mm)','p<.05','r_min']
        else:
            return ['%s'%self.tag.replace('_','/'),'%0.2f'%(1.0/np.max(np.abs(L))),'%s'%sig,'%0.5f'%r_min]
        
        
    def correct_duplicates(self):
        self.coneset.correct_duplicates()
        self.coneset.to_file(self.coordinate_fn)
        
    def has_coordinates_file(self):
        return os.path.exists(self.coordinate_fn)

    def get_densities_row(self,header=False):
        if header:
            return ['file_id','width (px)','height (px)','area (mm^-2)','count total','count s','count l/m','density total (mm^-2)','density s (mm^-2)','density l/m (mm^-2)']
        else:
            area_sqmm = self.area_sqmm
            total_count = self.s_count + self.lm_count
            total_density = total_count/area_sqmm
            lm_density,s_density = float(self.lm_count)/area_sqmm,float(self.s_count)/area_sqmm
            return [self.tag.replace('_','/'),self.sx,self.sy,'%0.5f'%area_sqmm,total_count,self.s_count,self.lm_count,'%0.1f'%total_density,'%0.1f'%s_density,'%0.1f'%lm_density]

    def do_counts(self):
        self.s_count = 0
        self.lm_count = 0
        for cone in self.coneset.cones:
            if cone.label.lower()=='s':
                self.s_count = self.s_count + 1
            elif cone.label.lower()=='l/m':
                self.lm_count = self.lm_count + 1

    def get_s_voronoi_statistics_row(self,header=False):
        cs = self.coneset
        rs = cs.random_set()
        cs.make_regions()
        rs.make_regions()
        hrow = ['file_id','N(S-cones)','real area mean','random area mean','real area variance','random area variance','real area std','random area std','real area max','random area max','real area min','random area min','real area median','random area median','bartlett T','bartlett p','levene W','levene p','fligner X','fligner p','f cdf','real roundness mean','random roundness mean','real roundness std','random roundness std','real c-c mean','random c-c mean','real c-c std','random c-c std','real nn mean','random nn mean','real nn std','random nn std']

        
        if header:
            row = hrow
        else:
            s_areas = np.array(cs.regions.get_area())*(self.mm_per_pixel**2)
            r_areas = np.array(rs.regions.get_area())*(self.mm_per_pixel**2)

            
            svar = np.var(s_areas)
            rvar = np.var(r_areas)
            sstd = np.std(s_areas)
            rstd = np.std(r_areas)
            smax = np.max(s_areas)
            smin = np.min(s_areas)
            smean = np.mean(s_areas)
            smed = np.median(s_areas)
            rmax = np.max(r_areas)
            rmin = np.min(r_areas)
            rmean = np.mean(r_areas)
            rmed = np.median(r_areas)
            b_T,b_p = bartlett(s_areas,r_areas)
            l_W,l_p = levene(s_areas,r_areas)
            f_X,f_p = fligner(s_areas,r_areas)
            cdf = ftest(len(s_areas),len(r_areas)).cdf(svar/rvar)

            sroundness = np.mean(cs.regions.get_roundness())
            rroundness = np.mean(rs.regions.get_roundness())
            sroundness_std = np.std(cs.regions.get_roundness())
            rroundness_std = np.std(rs.regions.get_roundness())

            sccvec = np.array(cs.regions.get_cc())*self.mm_per_pixel
            rccvec = np.array(rs.regions.get_cc())*self.mm_per_pixel
            
            scc = np.mean(sccvec)
            rcc = np.mean(rccvec)
            scc_std = np.std(sccvec)
            rcc_std = np.std(rccvec)

            s_nn = np.array(cs.regions.get_nn())
            r_nn = np.array(rs.regions.get_nn())
            
            snn = np.mean(s_nn)
            rnn = np.mean(r_nn)
            snn_std = np.std(s_nn)
            rnn_std = np.std(r_nn)
            
            
            formats = ['%s','%d','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e','%0.3e']
            data = [self.tag.replace('_','/'),len(self.coneset),smean,rmean,svar,rvar,sstd,rstd,smax,rmax,smin,rmin,smed,rmed,b_T,b_p,l_W,l_p,f_X,f_p,cdf,sroundness,rroundness,sroundness_std,rroundness_std,scc,rcc,scc_std,rcc_std,snn,rnn,snn_std,rnn_std]
            row =  []
            for fmt,dat,h in zip(formats,data,hrow):
                row.append(fmt%dat)
        return row
            
    def get_s_nearest_neighbor_statistics_row(self,header=False):
        if header:
            row = ['file_id','lambda','E(NND_CSR) (mm)','N(S-cones)','N(S-cones with NND>E(NND_CSR))',
                   'P(NND_S>NND_CSR)','mean(CDF_CSR)']
        else:


            N = len(self.s_coneset)
            L = float(N)/self.area_sqmm

            d_csr = np.sqrt(-np.log(0.5)/L/np.pi)

            nnd,nn = self.s_coneset.get_nearest_neighbors()
            nnd = nnd * self.mm_per_pixel

            N_greater_d_csr = len(np.where(nnd>d_csr)[0])

            p_greater_d_csr = float(N_greater_d_csr)/float(N)
            
            drange = np.linspace(0,np.max(nnd),1024)
            fd = 1-np.exp(-L*np.pi*drange**2)
            cdf = 1 - np.exp(-L*np.pi*nnd**2)

            mcdf = np.mean(cdf)

            row = [self.tag.replace('_','/'),'%0.1f'%L,'%0.5f'%d_csr,'%d'%N,'%d'%N_greater_d_csr,'%0.3f'%p_greater_d_csr,'%0.6f'%mcdf]

            outd,outfn = os.path.split(self.tif_fn.replace('images','s_nearest_neighbors').replace('.tif','_s_nn.csv'))
            if not os.path.exists(outd):
                os.makedirs(outd)
            outfn = os.path.join(outd,outfn)
            with open(outfn,'wb') as fid:
                writer = csv.writer(fid)
                writer.writerow(['cone_x','cone_y','neighbor_x','neighbor_y','d (mm)','d (px)'])
                for cone,neighbor,distance in zip(self.s_coneset.cones,nn,nnd):
                    grow = ['%0.2f'%cone.x,'%0.2f'%cone.y,'%0.2f'%neighbor.x,'%0.2f'%neighbor.y,'%0.5f'%distance,'%0.2f'%(distance/self.mm_per_pixel)]
                    writer.writerow(grow)
        return row
        

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

    def clear_below(self,frac,filt='l/m,s',rad=2):
        gmax = np.max(self.gbs)
        gmin = np.min(self.gbs)
        grange = gmax - gmin
        gthresh = gmin + frac * grange
        
        rmax = np.max(self.rbs)
        rmin = np.min(self.rbs)
        rrange = rmax - rmin
        rthresh = rmin + frac * rrange

        newcones = []
        for cone in self.coneset.cones:
            if filt.find(cone.label)>-1:
                x,y = round(cone.x),round(cone.y)
                x1,y1 = max(x-rad,0),max(y-rad,0)
                x2,y2 = min(x+rad+1,self.sx),min(y+rad+1,self.sy)
                
                if cone.label=='l/m':
                    test = np.mean(self.rbs[y1:y2,x1:x2])
                    print x1,x2,y1,y2,test,self.r.shape
                    if test>rthresh:
                        newcones.append(cone)
                elif cone.label=='s':
                    test = np.mean(self.gbs[y1:y2,x1:x2])
                    print x1,x2,y1,y2,test,self.g.shape
                    if test>gthresh:
                        newcones.append(cone)
        self.coneset.cones = newcones
        self.coneset.update()
                
        
        
    def load_cones(self):
        self.coneset.from_file(self.coordinate_fn)
        
    def imshow(self,dpi=50,fn=None):
        plt.figure(figsize=(self.sx/dpi,self.sy/dpi))
        ax = plt.axes([0,0,1,1])
        plt.imshow(self.im0)
        plt.xticks([])
        plt.yticks([])
        self.coneset.plot(ax)
        if fn is not None:
            print 'saving marked image to %s'%fn
            plt.savefig(fn,dpi=dpi)
        
        
    def filter(self,sigma,frac,nbins,erosion_diameter):
        self.gto = threshold(self.g,sigma,frac,nbins,erosion_diameter)
        self.rto = threshold(self.r,sigma,frac,nbins,erosion_diameter)
        self.filtered = high_contrast(self.rto,self.gto)
        return self.filtered
        
    def autodetect_peaks(self,sigma,frac,nbins,erosion_diameter):
        self.gto = threshold(self.g,sigma,frac,nbins,erosion_diameter)
        self.rto = threshold(self.r,sigma,frac,nbins,erosion_diameter)
        self.filtered = high_contrast(self.rto,self.gto)
        g_mask = np.zeros(self.gto.shape)
        g_mask[np.where(self.gto)] = 1.0
        g_mask = morphology.binary_dilation(g_mask,structure=strel(diameter=5))
        r_mask = np.zeros(self.rto.shape)
        r_mask[np.where(self.rto)] = 1.0
        r_mask = morphology.binary_dilation(r_mask,structure=strel(diameter=5))

        rx,ry = centroid_objects(self.r,r_mask)
        gx,gy = centroid_objects(self.g,g_mask)

        self.coneset.clear()
        for x,y in zip(rx,ry):
            self.coneset.add(x,y,'l/m')
        for x,y in zip(gx,gy):
            self.coneset.add(x,y,'s')
        

    def autodetect_peaks_alt(self,sigma,neighborhood_size):
        kernel_size = 11
        data = gaussian_blur(self.im,sigma,kernel_size)
        data_max = filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)

        labeled, num_objects = ndimage.label(maxima)

        slices = ndimage.find_objects(labeled)
        x, y = [], []

        for dy,dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            x.append(x_center)
            y_center = (dy.start + dy.stop - 1)/2    
            y.append(y_center)


        gmr = self.g-self.r
        self.coneset.clear()
        for idx,(xi,yi) in enumerate(zip(x,y)):
            if gmr[yi,xi]>0:
                self.coneset.add(xi,yi,'s')
            else:
                self.coneset.add(xi,yi,'l/m')


class ConeSet:

    id = 0
    def __init__(self,mask):
        self.cones = []
        self.update()
        self.id = ConeSet.id
        self.mask = mask
        ConeSet.id += 1

    def __str__(self):
        return 'ConeSet %d with %d cones: %s.'%(self.id,len(self.cones),self.cones)
        #return self.cones.__str__()
        
    def __repr__(self):
        return 'ConeSet %d with %d cones: %s.'%(self.id,len(self.cones),self.cones)
        #return self.cones.__repr__()

    def __add__(self,other):
        out = ConeSet()
        out.cones = self.cones + other.cones
        out.update()
        return out

    def __len__(self):
        return len(self.cones)

    def random_set(self):
        coneset = ConeSet(self.mask)
        sy,sx = self.mask.shape
        k = 0
        while k< len(self.cones):
            x,y = np.random.rand()*(sx-1),np.random.rand()*(sy-1)
            if self.mask[round(y),round(x)]:
                coneset.add(x,y,self.labels[k])
                k = k + 1
        return coneset

    def get_nearest_neighbors(self):
        nearest_neighbors = []
        nearest_distances = []
        for cone in self.cones:
            candidates = []
            distances = []
            for candidate in self.cones:
                if cone!=candidate:
                    candidates.append(candidate)
                    distances.append(cone.distance(candidate))
            idx = np.argmin(distances)
            nearest_distances.append(distances[idx])
            nearest_neighbors.append(candidates[idx])
        return np.array(nearest_distances),np.array(nearest_neighbors)
                    

    def make_distance_matrix(self,fn,mm_per_pixel):
        N = len(self.cones)
        with open(fn,'wb') as fid:
            writer = csv.writer(fid)
            formats = ['%0.5f']*N
            for n in range(N):
                row = [0.0]*N
                for m in range(n+1,N):
                    row[m]=self.cones[n].distance(self.cones[m])*mm_per_pixel

                printrow = [fmt%dat for fmt,dat in zip(formats,row)]

                writer.writerow(printrow)
                print printrow
        
    
    def correct_duplicates(self):
        r = np.round(self.xvec) + np.round(self.yvec)*1j
        u,indices = np.unique(r,return_index=True)
        newcones = []
        for idx in indices:
            newcones.append(self.cones[idx])
        self.cones = newcones
        self.update()
    
    def check_duplicates(self):
        r = self.xvec + self.yvec*1j
        ulen = len(np.unique(r))
        print len(r),ulen
        possible_duplicates = len(r)>ulen
        if possible_duplicates:
            self.log('ConeSet may contain duplicates.')
        return possible_duplicates
    
    def log(self,msg):
        print msg
    
    def update(self):
        self.xvec = np.array([c.x for c in self.cones])
        self.yvec = np.array([c.y for c in self.cones])
        self.labels = [c.label for c in self.cones]

    def clear(self,filt='all'):
        newcones = []
        for cone in self.cones:
            if cone.label==filt or filt=='all':
                pass
            else:
                newcones.append(cone)
        self.cones = newcones
        self.update()
        self.log('Cleared %s. Result: %s.'%(filt,self))
        
    def edit(self,x,y,label):
        idx = self.remove(x,y,tolerance=5.0,label=label)
        if idx==-1:
            self.add(x,y,label)
        
    def remove(self,x,y,tolerance=5.0,label=None):
        if len(self.cones)==0:
            return -1
        d = np.sqrt((x-self.xvec)**2+(y-self.yvec)**2)
        idx = np.argmin(d)
        dmin = d[idx]
        if dmin<tolerance and (label is None or label==self.labels[idx]):
            self.cones = self.cones[:idx]+self.cones[idx+1:]
            self.update()
            return idx
        else:
            return -1

    def add(self,x,y,label):
        self.append(Cone(x,y,label))

    def append(self,cone):
        self.cones.append(cone)
        self.update()

    def to_file(self,fn,delimiter=','):
        if len(self.cones)==0:
            self.log('No cones to write to file %s.'%fn)
            return
        with open(fn,'w') as csvfile:
            writer = csv.writer(csvfile,delimiter=delimiter)
            writer.writerows(zip(self.xvec,self.yvec,self.labels))

    def from_file(self,fn,delimiter=',',filt=None):
        self.cones = []
        count = 0
        with open(fn,'rb') as csvfile:
            reader = csv.reader(csvfile,delimiter=delimiter)
            for row in reader:
                if filt is None or row[2]==filt:
                    self.cones.append(Cone(float(row[0]),float(row[1]),row[2]))
                    count = count + 1
        self.update()
        if filt is None:
            fstr = 'any'
        else:
            fstr = filt
        self.log('Loaded %d cone coordinates from %s with %s label.'%(count,fn,fstr))
    
    def plot_old(self,axes):
        axes.autoscale(False)
        for x,y,label in zip(self.xvec,self.yvec,self.labels):
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

                
    def plot(self):
        plt.autoscale(False)
        lm_idx = [idx for idx,label in enumerate(self.labels) if label=='l/m']
        s_idx = [idx for idx,label in enumerate(self.labels) if label=='s']
        l_x,l_y = self.xvec[lm_idx],self.yvec[lm_idx]
        s_x,s_y = self.xvec[s_idx],self.yvec[s_idx]
        
        ph = plt.plot(l_x,l_y,'ro')
        ph[0].set_markeredgecolor('k')
        ph[0].set_markerfacecolor('y')
        ph[0].set_alpha(.7)
        ph[0].set_markersize(5)
        ph = plt.plot(s_x,s_y,'gs')
        ph[0].set_markeredgecolor('k')
        ph[0].set_markerfacecolor('b')
        ph[0].set_markersize(5)
        ph[0].set_alpha(.7)

    def plot_on(self,axes):
        axes.autoscale(False)
        lm_idx = [idx for idx,label in enumerate(self.labels) if label=='l/m']
        s_idx = [idx for idx,label in enumerate(self.labels) if label=='s']
        l_x,l_y = self.xvec[lm_idx],self.yvec[lm_idx]
        s_x,s_y = self.xvec[s_idx],self.yvec[s_idx]
        
        ph = axes.plot(l_x,l_y,'ro')
        ph[0].set_markeredgecolor('k')
        ph[0].set_markerfacecolor('y')
        ph[0].set_alpha(.7)
        ph[0].set_markersize(5)
        ph = axes.plot(s_x,s_y,'gs')
        ph[0].set_markeredgecolor('k')
        ph[0].set_markerfacecolor('b')
        ph[0].set_markersize(5)
        ph[0].set_alpha(.7)

    def make_regions(self,label=None):
        self.regions = RegionSet(self.xvec,self.yvec,self.mask)
        

class RegionSet:
    
    def __init__(self,x,y,mask,label='no_label'):
        """ The mask is a binary 2D array containing 1s in valid regions and 0s elsewhere. """
        self.label = label
        self.items = []
        #print 'Starting with %d items.'%(len(self.items))
        xy = np.vstack((x,y)).T
        vor_in = Voronoi(xy)
        count = 0

        xmin,ymin = 0,0
        ymax,xmax = mask.shape
        
        for idx,point in enumerate(vor_in.points):
            vregionidx = vor_in.point_region[idx]
            region_vertices = vor_in.regions[vregionidx]
            if len(region_vertices)>2 and np.min(region_vertices)>=0:
                region_Points = []
                degenerate = False
                for region_vertex in region_vertices:
                    vtx = vor_in.vertices[region_vertex]
                    mask_ytemp = int(np.fix(vtx[1]))
                    mask_xtemp = int(np.fix(vtx[0]))
                    # original line: if (vtx[0]>xmin and vtx[0]<xmax and vtx[1]>ymin and vtx[1]<ymax and mask[np.fix(vtx[1]),np.fix(vtx[0])]):
                    if (vtx[0]>xmin and vtx[0]<xmax and vtx[1]>ymin and vtx[1]<ymax and mask[mask_ytemp,mask_xtemp]):
                        region_Points.append(geometry.Point(vtx[0],vtx[1]))
                    else:
                        degenerate = True
                        break
                if not degenerate:
                    region = geometry.Region(geometry.Point(point[0],point[1]),region_Points)
                    self.append(region)
                    count = count + 1
        #print 'Added %d items.'%(count)
        #print 'Have %d items.'%(len(self.items))
        
    def append(self,item):
        self.check(item)
        self.items.append(item)

    def check(self,item):
        if not isinstance(item,geometry.Region):
            raise Exception('You cannot add a %s to a RegionSet.'%(type(item)))

    def __len__(self):
        return len(self.items)
        
    def get_att(self,attname):
        out = []
        for item in self.items:
            out.append(eval('item.get_%s()'%attname))
        return out
    
    def get_roundness(self):
        out = []
        for item in self.items:
            out.append(item.get_roundness())
        return out

    def get_nn(self):
        out = []
        for item in self.items:
            out.append(item.get_nn())
        return out
    
    def get_cc(self):
        out = []
        for item in self.items:
            out.append(item.get_cc())
        return out

    def get_area(self):
        out = []
        for item in self.items:
            out.append(item.get_area())
        return out

    def get_nn(self):
        out = []
        for item in self.items:
            out.append(item.get_nn())
        return out
        
#    def add_voronoi(self,vor_in,xmin,xmax,ymin,ymax):

    def plot(self,marker=DEFAULT_MARKER,plot_type=None):
        for r in self.items:
            r.plot(marker,plot_type)


    def get_regularity_index(self,mode='area'):
        if mode=='area':
            vals = np.array(self.get_area())
        elif mode=='radius':
            vals = np.sqrt(np.array(self.get_area())/np.pi)

        q75, q25 = np.percentile(vals, [75 ,25])
        iqr = q75 - q25
        bin_size = 2*iqr/(float(len(vals))**(1.0/3.0))
        
        nbins = int(np.ceil((np.max(vals)-np.min(vals))/bin_size))

        ri = np.mean(vals)/np.std(vals)

        return ri,vals,nbins
        

    def make_regularity_table(self,fn):
        avals = np.array(self.get_area())
        rvals = np.sqrt(avals/np.pi)

        plt.figure()
        for idx,r in enumerate(self.items):
            r.plot(plot_type=idx)

        ax = plt.gca()
        plt.axis('image')
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.xticks([])
        plt.yticks([])
        
        plt.savefig(fn+'_regularity_table_map.png')
        plt.close()

        outfid = open(fn+'_regularity_table.csv','w')
        outfid.write('index,area,radius,\n')
        for idx,(a,r) in enumerate(zip(avals,rvals)):
            outfid.write('%04d,%0.1f,%0.1f,\n'%(idx,a,r))
        outfid.close()

    
        
class HexGrid(RegionSet):

    #def __init__(self,xmin,xmax,ymin,ymax,area):
    def __init__(self,mask):
        self.area = np.sum(mask)
        dx = mask.shape[1]
        dy = mask.shape[0]
        
        self.xmin = 0
        self.ymin = 0
        self.xmax = dx
        self.ymax = dy

        long_radius = np.sqrt(2*float(self.area)/3/np.sqrt(3))
        short_radius = long_radius*np.cos(np.pi/6.0)
        x = self.xmin + short_radius
        y = self.ymin + long_radius
        xend = self.xmax - short_radius
        yend = self.ymax - long_radius

        rowidx = 0
        xs = []
        ys = []
        while y<=yend:
            while x<=xend:
                if x>=self.xmin+short_radius and x<=self.xmax-short_radius and y>=self.ymin+long_radius and y<=self.ymax-long_radius:
                    xs.append(x)
                    ys.append(y)
                x = x + 2*short_radius
            x = short_radius * (rowidx%2)
            y = y + long_radius*1.5
            rowidx = rowidx + 1

        self.xs = xs
        self.ys = ys
        self.xs0 = xs
        self.ys0 = ys

        RegionSet.__init__(self,self.xs,self.ys,mask)


        
        self.short_radius = short_radius
        self.long_radius = long_radius
        self.step_size = long_radius * .15
        self.n = len(xs)

#8-430
#6475400
