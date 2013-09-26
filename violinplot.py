from matplotlib.pyplot import figure, show, gca
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from numpy.random import normal
from numpy import arange
import random
import numpy as np
import scipy.stats
import scipy.stats.mstats
import scipy.signal
from matplotlib.patches import Circle, Wedge, Polygon, PathPatch
from matplotlib.path import Path
from matplotlib.collections import PatchCollection

def stripe(x,p,v,left):
    stripes = np.linspace(0, len(x), 30)
    patches = []
    for s in stripes:
        ix = int(s)-1        
        if left:
            xs = np.linspace(p,v[ix]+p, 10)    
        else:
            xs = np.linspace(p,-v[ix]+p, 10)    
        
        fn = v[ix:min(ix+10, len(v)-1)] - v[ix]
        
        
        if len(fn)==len(xs):
            ys = x[ix] - fn             
            dx = np.gradient(xs)
            dy = np.gradient(ys)
            n = np.sqrt(dx**2+dy**2)
            dx = dx/n
            dy = dy/n
            dz = dx*dy
            n = np.sqrt(dx**2+dy**2+dz**2)
            dx = dx/n
            dy = dy/n
            dz = -dz/n
            if left:
                dx = -dx        
                
            w = (dz) * 0.2
                    
            ytop = ys
            ybot = ys - w
            
            tops = np.column_stack((xs, ytop))
            bots = np.column_stack((xs, ybot))        
            
            outline = np.vstack((tops, bots[::-1])) 
            patches.append(Polygon(outline, True))     
    p = PatchCollection(patches, edgecolor='none', facecolor='black', alpha=0.2)        
    ax.add_collection(p)
        
def violin(d,x,p,v,w,left=True, mean=False, mode=False, outliers=False, normal=False, lcolor='y', rcolor='y', outlier_symbol='kx', non_normal=False):

    if left:
        ax.fill_betweenx(x,p,v+p,facecolor=rcolor,alpha=0.2,lw=2)
    else:
        ax.fill_betweenx(x,p,-v+p,facecolor=lcolor,alpha=0.2,lw=2)
        
    #stripe(x,p,v,left)
        
        
    mn = np.mean(d)
    st = np.std(d)
    sem = scipy.stats.sem(d)  * 1.96      
    six = ((x>mn-sem) & (x<mn+sem))
        
    vl, l25, med, u25, vh = scipy.stats.mstats.mquantiles(d, prob=[0.02, 0.25, 0.5, 0.75, 0.98])
    
    if outliers:
        out_high = d[d>vh]
        out_low = d[d<vl]        
        if left:
            ax.plot([p+w/3 for i in range(len(out_high))],out_high, outlier_symbol, alpha=0.4)
            ax.plot([p+w/3 for i in range(len(out_low))],out_low, outlier_symbol, alpha=0.4)
        else:
            ax.plot([p-w/3 for i in range(len(out_high))],out_high, outlier_symbol, alpha=0.4)
            ax.plot([p-w/3 for i in range(len(out_low))],out_low, outlier_symbol, alpha=0.4)
        
    sthigh = ((x>med) & (x<u25))
    stlow = ((x<med) & (x>l25))
    
    if left:
        ax.fill_betweenx(x[sthigh],p,v[sthigh]+p,facecolor=rcolor,alpha=0.3,edgecolor='none')
    else:
        ax.fill_betweenx(x[sthigh],p,-v[sthigh]+p,facecolor=lcolor,alpha=0.3,edgecolor='none')
        
    if left:
        ax.fill_betweenx(x[stlow],p,v[stlow]+p,facecolor=rcolor,alpha=0.3,edgecolor='none')
    else:
        ax.fill_betweenx(x[stlow],p,-v[stlow]+p,facecolor=lcolor,alpha=0.3,edgecolor='none')
         
    
    med_i = np.argmin(np.abs((x-med)))
    med_v = v[med_i]
    
    if left:
        ax.plot([p, p+med_v], [med, med], color=rcolor, lw=2)
        ax.plot([p, p+w/3], [vl, vl], 'k', lw=2, alpha=0.5)
        ax.plot([p, p+w/3], [vh, vh], 'k', lw=2, alpha=0.5)
        
    else:
        ax.plot([p-med_v, p], [med, med], color=lcolor, lw=2)
        ax.plot([p, p-w/3], [vl, vl], 'k', lw=2, alpha=0.5)
        ax.plot([p, p-w/3], [vh, vh], 'k', lw=2, alpha=0.5)
        
       
    if mean:
        
        w2 = w / 1.648
        w3 = w/3
        w4 = w / 1.401
        if left:
            ax.plot([p, p+w], [mn, mn], 'g', lw=2)            
            ax.plot([p, p+w2], [mn+st, mn+st], 'g')                        
            ax.plot([p, p+w2], [mn-st, mn-st], 'g')
            ax.plot([p, p+w3], [mn+st*2.1, mn+st*2.1], 'g:')
            ax.plot([p, p+w3], [mn-st*2.1, mn-st*2.1], 'g:')
            ax.plot([p, p+w4], [mn+st*0.675, mn+st*0.675], 'g:')
            ax.plot([p, p+w4], [mn-st*0.675, mn-st*0.675], 'g:')
            
            
        else:
            ax.plot([p-w, p], [mn, mn], 'g',lw=2)
            ax.plot([p, p-w2], [mn+st, mn+st], 'g')                        
            ax.plot([p, p-w2], [mn-st, mn-st], 'g')
            ax.plot([p, p-w3], [mn+st*2.1, mn+st*2.1], 'g:')
            ax.plot([p, p-w3], [mn-st*2.1, mn-st*2.1], 'g:')
            ax.plot([p, p-w4], [mn+st*0.675, mn+st*0.675], 'g:')
            ax.plot([p, p-w4], [mn-st*0.675, mn-st*0.675], 'g:')
        
    gx = np.exp(-((x-mn)**2)/(2*st**2)) * w
    if normal:
        if left:
            ax.fill_betweenx(x, p+gx, p, facecolor='none', alpha=0.3, edgecolor='g', lw=2)
        else:
            ax.fill_betweenx(x, p-gx, p, facecolor='none', alpha=0.3, edgecolor='g', lw=2)
            
        if left:
            ax.fill_betweenx(x[six],p,gx[six]+p,facecolor='g',alpha=0.3,edgecolor='none')
        else:
            ax.fill_betweenx(x[six],p,-gx[six]+p,facecolor='g',alpha=0.3,edgecolor='none')
    
    
    if non_normal:
        if left:
            ax.fill_betweenx(x, p+v, p+gx, facecolor='r', alpha=0.2, edgecolor='none', lw=0)
        else:
            ax.fill_betweenx(x, p-gx, p-v, facecolor='r', alpha=0.2, edgecolor='none', lw=0)
            
        
        
                                                                                                                   
    if mode:
        maxima = (np.diff(np.sign(np.diff(v))) < 0).nonzero()[0] + 1
        maxima = [m for m in maxima if v[m]>1e-2]
        mx, my = v[maxima], x[maxima]
        if left:
            ax.plot(p+mx+0.02, my, 'k<', alpha=0.4)
        else:
            ax.plot(p-mx-0.02, my, 'k>', alpha=0.4)
    

def violin_side(d,p,w, kde=None, **kwargs):
    k = gaussian_kde(d, kde) #calculates the kernel density
    
    vl, vh = scipy.stats.mstats.mquantiles(d, prob=[0.001, 0.999])
    vl, vh = k.dataset.min(), k.dataset.max()
    m = vl
    M = vh
    x = arange(m,M,(M-m)/100.) # support for violin
    v = k.evaluate(x) #violin profile (density curve)
    v = v/v.max()*w #scaling the violin to the available space
    violin(d,x,p,v,w,**kwargs)
    
        
def regression(pos, data, color='r'):
    xs = []
    ys = []
    for p,d in zip(pos, data):
        ps = np.ones((len(d),)) * p
        xs.append(ps)
        ys.append(d)            
    xs = np.hstack(xs)
    ys = np.hstack(ys)        
    slope,intercept,r,p,err = scipy.stats.linregress(xs,ys)
    ab = np.linspace(min(pos) - (pos[1]-pos[0]), max(pos) + (pos[-1]-pos[-2]), 100)
    ax.plot(ab, ab*slope+intercept, color, lw=3)

        
def violin_plot(data, pos=None, ax=None, alt_data=None, spacing=0.75,  regress=False, labels=None, label_rotate=0, **kwargs):
    '''
    Create violin plots on an axis. Shows distribution of the data use a kernel density estimate, along with median, interquartile ranges,
    2% and 98% percentiles. Can optionally show the mean, standard deviation, 95% CI and modes of the density estimate.
    
    data: Data to plot
    pos (optional): Positions of the violins. If None, uses 0,1,2,3... etc.
    ax (optional): axis to plot on to. If None, uses the current axis.
    alt_data (optional): alternative data to plot on the right side of the violin, for comparison. If present,
                         must be the same length as data.
    spacing (optional): Spacing factor. 1.0 = fill available width, 0.5=half available width, etc.
    regress (optional): If True, draw linear regression lines
    lcolor (optional): Color of the left hand side of the graph
    rcolor (optional): Color of the right hand side of the graph
    mean (optional): If True, show the mean, standard deviation and standard error interval.
    mode (optional): If True, show the modes of the distrbution
    kde: (optional): Sets the kernel density estimate parameters. Can be scalar width (e.g. 0.1), 
                    or automatic methods 'silverman' or 'scott'. See scipy.stats.gaussian_kde() for details
    normal: (optional): If True, show the best fit normal approximation to the data
    non_normal (optional): If True, show difference between distribution and the normal approximation.
    outliers: (optional): If True, show outlier symbols
    outlier_symbol (optional): Symbol to use for outliers. Default is 'kx'    
    labels (optional): Label for each position. If given must be the same length as data.
    label_rotate (optional): Rotation of labels, in degrees. Default is 0
       
    '''
    if not ax:
        ax = gca()
        
    if not pos:
        pos = np.arange(len(data))
        
    dist = max(pos)-min(pos)
    w = min(0.15*max(dist,1.0),0.5)
    w = w * spacing
    
    if alt_data == None:
        alt_data =  data
        
        
    for d,bd,p in zip(data,alt_data, pos):
        
        violin_side(d,p,w,left=False,**kwargs)
        violin_side(bd,p,w,left=True,**kwargs)
        
    if regress:
        regression(pos, data, 'k:')
        regression(pos, alt_data, 'k--')
        
    if labels:        
        #
        #
        ax.set_xticklabels(['']+labels)
        locs, labels = plt.xticks()
        if label_rotate:
            plt.setp(labels, rotation=label_rotate)
        
        
def mix(a,b):
    return np.array([random.choice([x,y]) for x,y in zip(a,b)])
    
if __name__=="__main__":
    pos = range(5)
    data = [normal(i/2,1,size=150) for i in pos]
    data2 = [mix(normal(2,0.1,size=200), normal(-2,1,size=200)) for i in pos]
            
    fig=figure()
    ax = fig.add_subplot(111)

    violin_plot(data,pos=pos,alt_data=data2,mean=True,mode=True,normal=True, regress=False, lcolor=[0.9,0.7,0.1], rcolor=[0.2,0.8,0.6], kde=0.5, non_normal=True, labels=['one', 'two', 'three', 'four', 'five'])
    show()