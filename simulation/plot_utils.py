import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_1Dhist(
        fig, 
        data, 
        bins      = 100, 
        xlab      = "x", 
        ylab      = "y", 
        title     = "Title",
        ax        = None,
        ax_id     = 1,
        nrows     = 1,
        ncols     = 1,
        fontsize  = 14,
        histtype  = "stepfilled", 
        edgecolor = "#06416D", 
        facecolor = "#7eb0d5",
        label     = "data",
        legend    = True,
    ):
    
    # Set the font family
    plt.rcParams["font.family"] = "serif"
    
    # Check if an axis was passed to the function. If not, create one.
    if ax is None:
        ax = fig.add_subplot(nrows, ncols, ax_id)
        
    # Plot the histogram
    ax.hist(
        data, 
        bins      = bins,
        histtype  = histtype,
        edgecolor = edgecolor,
        facecolor = facecolor,
        label     = label,    
    )
    
    # If the legend flag is set to True, create a legend
    if legend:
        ax.legend(prop={'size': fontsize, "family": "serif"})
    
    # Set the x and y labels
    ax.set_xlabel(xlab, fontsize=fontsize, fontname="serif")
    ax.set_ylabel(ylab, fontsize=fontsize, fontname="serif")
    
    # Set the plot title
    ax.set_title(title, fontsize=fontsize+2, fontname="serif")
    
    # Set the tick parameters
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=5)
    
    # Return the axis
    return ax



def display_statistics(data, ax, fontsize=16, align="left", x=0.05, y=0.95, step=0.05):
    
    xmin   = np.min(data)
    xmax   = np.max(data)
    mean   = np.mean(data)
    std    = np.std(data)
    median = np.median(data)
    mode   = data.mode()[0]
    
    ax.text(x, y-0*step, f"Minimum: {xmin:.2f}",  fontsize=fontsize, transform=ax.transAxes, ha="left")
    ax.text(x, y-1*step, f"Maximum: {xmax:.2f}",  fontsize=fontsize, transform=ax.transAxes, ha="left")
    ax.text(x, y-2*step, f"Mean: {mean:.2f}",     fontsize=fontsize, transform=ax.transAxes, ha="left")
    ax.text(x, y-3*step, f"Std: {std:.2f}",       fontsize=fontsize, transform=ax.transAxes, ha="left")
    ax.text(x, y-4*step, f"Median: {median:.2f}", fontsize=fontsize, transform=ax.transAxes, ha="left")
    ax.text(x, y-5*step, f"Mode: {mode:.2f}",     fontsize=fontsize, transform=ax.transAxes, ha="left")
    
    return ax



def plot_1Dstack(
        fig, 
        data,
        hue, 
        bins         = 100, 
        xlab         = "x", 
        ylab         = "y", 
        title        = "Title",
        ax           = None,
        ax_id        = 1,
        nrows        = 1,
        ncols        = 1,
        fontsize     = 14,
        legend       = True,   
        legend_title = "Particle",
        legend_out   = True,
    ):
    
    # Set the font family
    plt.rcParams["font.family"] = "serif"
    
    # Check if an axis was passed to the function. If not, create one.
    if ax is None:
        ax = fig.add_subplot(nrows, ncols, ax_id)
    
    # create a temporary dataframe with two columns "data" and "hue"
    temp_df = pd.DataFrame({"data": data, "hue": hue})

    # sort the dataframe by particle abunance i.e. how many particles of each type
    hue_ordering = temp_df.groupby("hue").count().sort_values(by="data", ascending=False).index

    g = sns.histplot(
        data      = temp_df, 
        x         = "data", 
        bins      = bins, 
        hue       = "hue",
        hue_order = hue_ordering,
        ax        = ax, 
        element   = "step",
        multiple  = "stack"
    )
    
    if legend:
        if legend_out:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        g.legend_.set_title(legend_title, prop={"size": fontsize, "family": "serif"})
        # chage fontsize of the legend
        for text in g.legend_.get_texts():
            text.set_fontsize(fontsize)
            text.set_fontname("serif")
    
    # Set the x and y labels
    ax.set_xlabel(xlab, fontsize=fontsize, fontname="serif")
    ax.set_ylabel(ylab, fontsize=fontsize, fontname="serif")
    
    # Set the plot title
    ax.set_title(title, fontsize=fontsize+2, fontname="serif")
    
    # Set the tick parameters
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=5)
    
    # Return the axis
    return ax