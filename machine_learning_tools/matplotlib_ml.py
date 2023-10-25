'''



Notes on other functions: 
eventplot #will plot 1D data as lines, can stack multiple 1D events
-- if did a lot of these gives the characteristic neuron spikes
   all stacked on top of each other


matplot colors can be described with 
"C102" where C{number} --> there are only 10 possible colors
but the number can go as high as you want it just repeats after 10
Ex: C100  = C110

#How to set the figure size:
fig.set_size_inches(18.5, 10.5)

# not have the subplots run into each other
fig.tight_layout()








'''
from IPython.display import display
from copy import copy
from matplotlib import colors
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import webcolors


graph_color_list = ["blue","green","red","cyan","magenta",
     "black","grey","midnightblue","pink","crimson",
     "orange","olive","sandybrown","tan","gold","palegreen",
    "darkslategray","cadetblue","brown","forestgreen"]



def generate_random_color(print_flag=False,colors_to_omit=[]):
    if not nu.is_array_like(colors_to_omit):
        colors_to_omit = [colors_to_omit]
    current_color_list = [k for k in graph_color_list if k not in colors_to_omit]
    rand_color = np.random.choice(current_color_list,1)
    if print_flag:
        print(f"random color chosen = {rand_color}")
    return colors.to_rgba(rand_color[0])

def generate_unique_random_color_list(n_colors,print_flag=False,colors_to_omit=[]):
    total_colors = []
    
    for i in range(n_colors):
        found_color=False
        while not found_color:
            if not nu.is_array_like(colors_to_omit):
                colors_to_omit = [colors_to_omit]
            current_color_list = [k for k in graph_color_list if k not in colors_to_omit]
            rand_color = np.random.choice(current_color_list,1)
            if rand_color not in total_colors:
                if print_flag:
                    print(f"random color chosen = {rand_color}")
                total_colors.append(colors.to_rgba(rand_color[0]))
                found_color = True
    return total_colors
        
            
def generate_non_randon_named_color_list(n_colors,
                                        user_colors=[], #if user sends a prescribed list
                                        colors_to_omit=[],):
    """
    To generate a list of colors of a certain length 
    that is non-random
    
    """
    if n_colors <= 0:
        return []
    return mu.generate_color_list(n_colors = n_colors,
                                  user_colors=user_colors,
                                  colors_to_omit=colors_to_omit,
                      return_named_colors=True)
def generate_color_list(
                        user_colors=[], #if user sends a prescribed list
                        n_colors=-1,
                        colors_to_omit=[],
                        alpha_level=0.2,
                        return_named_colors = False):
    """
    Can specify the number of colors that you want
    Can specify colors that you don't want
    accept what alpha you want
    
    Example of how to use
    colors_array = generate_color_list(colors_to_omit=["green"])
    """
    #print(f"user_colors = {user_colors}")
    # if user_colors is defined then use that 
    if len(user_colors)>0:
        current_color_list = user_colors
    else:
        current_color_list = graph_color_list.copy()
    
    #remove any colors that shouldn't belong
    current_color_list = [k for k in current_color_list if k not in colors_to_omit]
    
    #print(f"current_color_list = {current_color_list}")
    
    if len(current_color_list) < len(user_colors):
        raise Exception(f"one of the colors you specified was part of unallowed colors {colors_to_omit}for a skeleton (because reserved for main)")
    
    #make a list as long as we need
    if n_colors > 0:
        current_color_list = (current_color_list*np.ceil(n_colors/len(current_color_list)).astype("int"))[:n_colors]
    
    if return_named_colors:
        return current_color_list
    
    #print(f"current_color_list = {current_color_list}")
    #now turn the color names all into rgb
    color_list_rgb = np.array([colors.to_rgba(k) for k in current_color_list])
    
    #changing the alpha level to the prescribed value
    color_list_rgb[:,3] = alpha_level
    
    return color_list_rgb


    
#----------------------------- Functions that were made for new graph visualization ------------------- #

def color_to_rgb(color_str):
    """
    To turn a string of a color into an RGB value
    
    Ex: color_to_rgb("red")
    """
    if type(color_str) == str:
        return colors.to_rgb(color_str)
    else:
        return np.array(color_str)

def color_to_rgba(current_color,alpha=0.2):
    curr_rgb = color_to_rgb(current_color)
    return apply_alpha_to_color_list(curr_rgb,alpha=alpha)
    
def get_graph_color_list():
    return copy(graph_color_list)

def generate_random_rgba(print_flag=False):
    rand_color = np.random.choice(graph_color_list,1)
    if print_flag:
        print(f"random color chosen = {rand_color}")
    return colors.to_rgb(rand_color[0])

def generate_color_list_no_alpha_change(
                        user_colors=[], #if user sends a prescribed list
                        n_colors=-1,
                        colors_to_omit=[],
                        alpha_level=0.2):
    """
    Can specify the number of colors that you want
    Can specify colors that you don't want
    accept what alpha you want
    
    Example of how to use
    colors_array = generate_color_list(colors_to_omit=["green"])
    """
    if len(user_colors)>0:
        current_color_list = user_colors
    else:
        current_color_list = graph_color_list.copy()
    
    if len(colors_to_omit) > 0:
        colors_to_omit_converted = np.array([color_to_rgb(k) for k in colors_to_omit])
        #print(f"colors_to_omit_converted = {colors_to_omit_converted}")
        colors_to_omit_converted = colors_to_omit_converted[:,:3]

        #remove any colors that shouldn't belong
        colors_to_omit = []
        current_color_list = [k for k in current_color_list if len(nu.matching_rows(colors_to_omit_converted,k[:3])) == 0]
    
    #print(f"current_color_list = {current_color_list}")
    
    if len(current_color_list) == 0:
        raise Exception(f"No colors remaining in color list after colors_to_omit applied ({current_color_list})")
    
    #make a list as long as we need

    current_color_list = (current_color_list*np.ceil(n_colors/len(current_color_list)).astype("int"))[:n_colors]
    
    return current_color_list


def process_non_dict_color_input(color_input):
    """
    Will return a color list that is as long as n_items
    based on a diverse set of options for how to specify colors
    
    - string
    - list of strings
    - 1D np.array
    - list of strings and 1D np.array
    - list of 1D np.array or 2D np.array
    
    *Warning: This will not be alpha corrected*
    """
    
    if color_input == "random": #if just string that says random
        graph_color_list = get_graph_color_list()
        color_list = [color_to_rgb(k) for k in graph_color_list]
    elif type(color_input) == str: #if just give a string then turn into list with string
        color_list = [color_to_rgb(color_input)]
    elif all(type(elem)==str for elem in color_input): #if just list of strings
        color_list = [color_to_rgb(k) for k in color_input]
    elif any(nu.is_array_like(elem) for elem in color_input): #if there is an array in the list 
        color_list = [color_to_rgb(k) if type(k)==str else k for k in  color_input]
    else:
        color_list = [color_input]
    
    return color_list

def apply_alpha_to_color_list(color_list,alpha=0.2,print_flag=False):
    single_input = False
    if not nu.is_array_like(color_list):
        color_list = [color_list]
        single_input = True
    color_list_alpha_fixed = []
    
    for c in color_list:
        if len(c) == 3:
            color_list_alpha_fixed.append(np.concatenate([c,[alpha]]))
        elif len(c) == 4:
            color_list_alpha_fixed.append(c)
        else:
            raise Exception(f"Found color that was not 3 or 4 length array in colors list: {c}")
    if print_flag:
        print(f"color_list_alpha_fixed = {color_list_alpha_fixed}")
    
    if single_input:
        return color_list_alpha_fixed[0]
    
    return color_list_alpha_fixed




def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def convert_rgb_to_name(rgb_value):
    """
    Example: convert_rgb_to_name(np.array([[1,0,0,0.5]]))
    """
    rgb_value = np.array(rgb_value)
    if not nu.is_array_like(rgb_value[0]):
        rgb_value = rgb_value.reshape(1,-1)
    
    #print(f"rgb_value.shape = {rgb_value.shape}")

    output_colors = []
    for k in rgb_value:
        if len(k) > 3:
            k = k[:3]
        adjusted_color_value = np.array(k)*255
        output_colors.append(get_colour_name(adjusted_color_value)[-1])
    
    if len(output_colors) == 1:
        return output_colors[0]
    elif len(output_colors) > 1:
        return output_colors
    else:
        raise Exception("len(output_colors) == 0")
        
def convert_dict_rgb_values_to_names(color_dict):
    """
    Purpose: To convert dictonary with colors as values to the color names
    instead of the rgb equivalents
    
    Application: can be used on the color dictionary returned by the 
    neuron plotting function
    
    Example: 
    from datasci_tools import matplotlib_utils as mu
    mu = reload(mu)
    nviz=reload(nviz)


    returned_color_dict = nviz.visualize_neuron(uncompressed_neuron,
                                                visualize_type=["network"],
                                                network_resolution="branch",
                                                network_directional=True,
                                                network_soma=["S1","S0"],
                                                network_soma_color = ["black","red"],       
                                                limb_branch_dict=dict(L1="all",
                                                L2="all"),
                                                node_size = 1,
                                                arrow_size = 1,
                                                return_color_dict=True)
                                                
    color_info = mu.convert_dict_rgb_values_to_names(returned_color_dict)
    
    
    """
    return dict([(k,convert_rgb_to_name(v)) for k,v in color_dict.items()])
    


base_colors_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


def plot_color_dict(colors,sorted_names=None, 
                    hue_sort=False,
                    ncols = 4,
                    figure_width = 20,
                    figure_height = 8,
                   print_flag=True):

    """
    Ex: 
    
    #how to plot the base colors
    Examples: 
    mu.plot_color_dict(mu.base_colors_dict,figure_height=20)
    mu.plot_color_dict(mu.base_colors_dict,hue_sort=True,figure_height=20)
    
    How to plot colors returned from the plotting function:
    from datasci_tools import matplotlib_utils as mu
    mu = reload(mu)
    nviz=reload(nviz)


    returned_color_dict = nviz.visualize_neuron(uncompressed_neuron,
                                                visualize_type=["network"],
                                                network_resolution="branch",
                                                network_directional=True,
                                                network_soma=["S1","S0"],
                                                network_soma_color = ["black","red"],       
                                                limb_branch_dict=dict(L1="all",
                                                L2="all"),
                                                node_size = 1,
                                                arrow_size = 1,
                                                return_color_dict=True)
                                                
    
    mu.plot_color_dict(returned_color_dict,hue_sort=False,figure_height=20)
    
    """
    if sorted_names is None:
        if hue_sort:
            # Sort colors by hue, saturation, value and then by name.
            by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                            for name, color in colors.items())
            #getting the names of the 
            sorted_names = [name for hsv, name in by_hsv]
        else:
            sorted_names = sorted(list(colors.keys()))
    n = len(sorted_names)
     #will always have 4 columns
    nrows = n // ncols + 1

    if print_flag:
        print(f"nrows = {nrows}")
        print(f"n-ncols*nrows = {n-ncols*nrows}")
    #creates figure
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    # Get height and width
    X, Y = fig.get_dpi() * fig.get_size_inches()
    h = Y / (nrows + 1)
    w = X / ncols

    for i, name in enumerate(sorted_names):
        row = i % nrows
        col = i // nrows
        y = Y - (row * h) - h

        xi_line = w * (col + 0.05)
        xf_line = w * (col + 0.25)
        xi_text = w * (col + 0.3)

        ax.text(xi_text, y, name, fontsize=(h * 0.5),
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y + h * 0.1, xi_line, xf_line,
                  #gets the color by name
                  color=colors[name], linewidth=(h * 0.6)
                 #color=[1,0,0,1],linewidth=(h * 0.6)
                 )

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1,
                        top=1, bottom=0,
                        hspace=0, wspace=0)
    plt.show()
    
    
    
# -------------------- Generic Plotting functions --------------------

"""
Best Idea for concatenating plotting is to create the figure 
first and then to pass the figure to different functions
and one of the subaxes will be altered within function


"""

def get_axes_locations_from_figure(fig):
    return [[f.get_subplotspec().colspan.start,
             f.get_subplotspec().rowspan.start] for f in fig.get_children()[1:]]
def get_axes_layout_from_figure(fig):
    return np.max(np.array(get_axes_locations_from_figure(fig)),axis=0) + np.array([1,1])

# plotting the BIC curve
def plot_graph(title,
                y_values,
                x_values,
                x_axis_label,
                y_axis_label,
              return_fig = False,
              figure = None,
              ax_index=None,
              label=None,
              x_axis_int=True):
    """
    Purpose: For easy plotting and concatenating plots
    """
    if figure is None:
        figure,ax = plt.subplots(1,1)
    else:
        ax = figure.axes[ax_index]
        return_fig = True
    
    
    ax.plot(x_values,y_values,label=label)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(title)
    if not label is None:
        ax.legend()
        
    if x_axis_int:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    if return_fig:
        plt.close()
        return figure
    else:
        plt.show()

def color_to_hex(color):
    return matplotlib.colors.to_hex(color, keep_alpha=False)
        
def display_figure(fig):
    display(fig)
    
# ---------- Helping with graph functions for network -------- #
def bins_from_width_range(bin_width,
                          bin_max,
                         bin_min=None):
    """
    To compute the width boundaries to help with 
    plotting and give a constant bin widht
    
    """
#     n_bins = (bin_max-bin_min)/bin_width
#     bins = np.arange(int(bin_min/bin_width),n_bins+1)*bin_width

    if bin_min is None:
        bin_min = 0
        
    return np.arange(bin_min,bin_max+0.00001,bin_width)


def histogram(data,
          n_bins=50,
          bin_width=None,
          bin_max=None,
          bin_min=None,
          density=False,
          logscale=False,
         return_fig_ax=True,
              fontsize_axes=20,
         **kwargs):
    """
    Ex: 
    histogram(in_degree,bin_max = 700,
         bin_width = 20,return_fig_ax=True)
    """
    
    if bin_width is not None and bin_max is not None:
        if bin_min is None:
            bin_min = 0
        bins=bins_from_width_range(bin_min =bin_min,
                                   bin_max=bin_max,
                                   bin_width=bin_width)
        
    else:
        bins=np.linspace(np.min(data),
                       np.max(data)+0.001,
                       n_bins)
        
    hist_heights,hist_bins = np.histogram(data, bins, density=density)

    x1 = hist_bins[:-1] #left edge
    x2 = hist_bins[1:] #right edge
    y = hist_heights
    w = np.array(x2) - np.array(x1) #variable width, can set this as a scalar also

    fig,ax = plt.subplots(1,1)
    ax.bar(x1, y, width=w, align='edge')

        
    ax.set_xlabel("Degree",fontsize=fontsize_axes)

    if not density:
        ax.set_ylabel("Count",fontsize=fontsize_axes)
    else:
        ax.set_ylabel("Density",fontsize=fontsize_axes)

    if logscale:
        ax.set_yscale("log")

    if not return_fig_ax:
        plt.show()
    else:
        return ax
    
def reset_default_settings():
    mpl.rcParams.update(mpl.rcParamsDefault)
    
def set_font_size(font_size):
    
    font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 20}
    
    matplotlib.rc('font', **font)
    
    
def add_random_color_for_missing_labels_in_dict(labels,
                                                label_color_dict,
                                               verbose = False):
    """
    Purpose: Will generate random colors for labels that
    are missing in the labels dict
    """
    unique_labels  = np.unique(labels.astype("str"))
    
    curr_keys = list(label_color_dict.keys())
    curr_colors = list(label_color_dict.values())
    labels_with_no_color = np.setdiff1d(unique_labels,curr_keys)
    
    if verbose:
        print(f"unique_labels = {unique_labels}")
        print(f"labels_with_no_color = {labels_with_no_color}")
        
    new_label_colors = mu.generate_non_randon_named_color_list(
                                    n_colors = len(labels_with_no_color),
                                    colors_to_omit=curr_colors)
    return_dict = label_color_dict.copy()
    for lab,col in zip(labels_with_no_color,new_label_colors):
        return_dict[lab] = col
        
    return return_dict

def set_legend_outside_plot(ax,scale_down=0.8):
    """
    Will adjust your axis so that the legend appears outside of the box
    """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * scale_down, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ax
    
def scatter_2D_with_labels(
    X,
    Y,
    labels,
    label_color_dict = None,
    x_title = "",
    y_title = "",
    axis_append = "",
    Z = None,
    z_title = "",
    alpha = 0.5,
    verbose = False,
    move_legend_outside_plot = True,
    ):
    """
    Purpose: Will plot scatter points
    where each point has a unique label
    (and allows to specify the colors of each label)

    Pseudocode: 
    1) Find the unique labels
    2) For all unique labels, if a color mapping is not 
    specified then add a random unique color (use function)

    3) Iterate through the labels to plot: 
    a. Find all indices of that label
    b. Plot them with the correct color and label
    
    4) Move the legend to outside of the plot
    
    mu.scatter_2D_with_labels(
    X = np.concatenate([f1_inh,f1_exc]),
    Y = np.concatenate([f2_inh,f2_exc]),
    #Z = np.ones(194),
    x_title = feature_1,
    y_title = feature_2,
    axis_append = "(per um of skeleton)",
    labels = np.concatenate([class_inh,class_exc]),
    alpha = 0.5,
    label_color_dict= dict(BC = "blue",
                        BPC = "black",
                        MC = "yellow",
                        excitatory = "red"
                    ),
    verbose = True)
    """
    if Z is not None:
        Z = np.array(Z)
        projection_type = "3d"
    else:
        projection_type = None
        
    X = np.array(X)
    Y = np.array(Y)

    if label_color_dict is None:
        label_color_dict = dict()
        

    labels= np.array(labels).astype("str")
    unique_labels = np.unique(labels)
    color_dict_adj = mu.add_random_color_for_missing_labels_in_dict(labels,
                                               label_color_dict,
                                               verbose = verbose)

    #fig,ax = plt.subplots(1,1,projection_type=projection_type)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = projection_type)
    
    for lab in unique_labels:
        lab_mask = labels == lab
        X_curr = X[lab_mask]
        Y_curr = Y[lab_mask]
        if Z is not None:
            Z_curr = Z[lab_mask] 
            ax.scatter(X_curr,Y_curr,Z_curr,c=color_dict_adj[lab],
                   label=lab,alpha = alpha)
        else:
            ax.scatter(X_curr,Y_curr,c=color_dict_adj[lab],
                   label=lab,alpha = alpha)

    ax.set_xlabel(f"{x_title} {axis_append}")
    ax.set_ylabel(f"{y_title} {axis_append}")
    if Z is not None:
        #ax.set_zlabel(f"{z_title} {axis_append}")
        ax.set_title(f"{z_title} vs {y_title} vs {x_title}")
    else:
        ax.set_title(f"{y_title} vs {x_title}")

    if move_legend_outside_plot:
        mu.set_legend_outside_plot(ax)
    else:
        ax.legend()

    plt.show()
    
  
    


#--- from machine_learning_tools ---
from . import numpy_ml as nu

from . import matplotlib_ml as mu