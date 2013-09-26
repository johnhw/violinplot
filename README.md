violinplot
==========

Matplotlib-based violin plots for Python. Originally based on the code at http://pyinsci.blogspot.co.uk/2009/09/violin-plot-with-matplotlib.html.

Usage:

    import violinplot
    # one set of data
    violinplot.violin_plot(data)
    
    # two sets of data (for comparison)
    violinplot.violin_plot(data, alt_data=data2)

    
    violin_plot(data, pos=None, ax=None, alt_data=None, spacing=0.75,  regress=False, labels=None, label_rotate=0, **kwargs):
    
Create violin plots on an axis. Shows distribution of the data use a kernel density estimate, along with median, interquartile ranges,
2% and 98% percentiles. Can optionally show the mean, standard deviation, standard error and modes of the density estimate.
    
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

