#importing all the libraries necessary.
import uproot
import math
import awkward as ak
import hist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sts
import matplotlib as mpl
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
from scipy.special import factorial
import mplhep as hep
import argparse
import os
import sys
import matplotlib.pylab as plab
from matplotlib.colors import ListedColormap
# everything in iminuit is done through the Minuit object, so we import it
from iminuit import Minuit
# we also need a cost function to fit and import the LeastSquares function
from iminuit.cost import LeastSquares
# display iminuit version
import iminuit
print("iminuit version:", iminuit.__version__)
#import zfit
#to have a standard graphic visualization.
#run the pycode from terminal.

#we define a series of functions that I need for fittings.
def gauss(x, A, mu, sigma): 
    return A/(np.sqrt(2*np.pi)*(sigma))*np.exp(- (x - mu)**2 /(2*(sigma**2)))

#perform the fit of an histogram with a gaussian function.
#input: data_frame, bin heights, bin borders, array of bound (A, mu, sigma) ([lowest values], [largest values])

def normal_fit(bin_heights, bin_borders, histo_name, bounds_array = ((-np.inf, -np.inf, -np.inf),(+np.inf, +np.inf, +np.inf)), init = [0, 0, 0.1], color_fit = "red"): 
    #calculate the bin centers, and then calculate the gauss function in the bin centers. 
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    least_squares = LeastSquares(bin_centers[bin_heights!=0], bin_heights[bin_heights!=0], np.sqrt(bin_heights[bin_heights!=0]), gauss)
    m = Minuit(least_squares, A=init[0], mu=init[1], sigma=init[2])  # starting values for α and β
    i = 0
    for p in m.parameters:
        m.limits[p] = (bounds_array[0][i], bounds_array[1][i])
        i += 1
    m.migrad()  # finds minimum of least_squares function
    m.hesse()   # accurately computes uncertainties
    # draw data and fitted line
    x_points = np.arange(bin_borders[0], bin_borders[-1], 0.01)
    histo_name.plot(x_points, gauss(x_points, *m.values), label="fit", color = color_fit)
    # display legend with some fit info
    fit_info = [""]
    #fit_info = [
    #    f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
    #]
    for p, v, e in zip(m.parameters, m.values, m.errors):
        if p == "A":
            fit_info.append(f"{p} = {v:.0f} $\\pm$ {e:.0f}")
        elif p == "mu":
            fit_info.append(f"$\mu$ = {v:.3f} $\\pm$ {e:.3f}")
        elif p == "sigma":
            fit_info.append(f"$\sigma$ = {v:.3f} $\\pm$ {e:.3f}")
    #### compute error on std
    histo_name.legend(title="\n".join(fit_info), frameon=False)
    return m

def fit_timing_gauss(timing_data, timing_bins, timing_range):
    obs = zfit.Space("x", limits=timing_range)

    zfit.Parameter._existing_params.clear()
    mu_init = np.mean(timing_data)
    sigma_init = np.std(timing_data)
    try:
        """ Define fit parameters: """
        mu = zfit.Parameter("mu" , mu_init, 0, 16)
        sigma = zfit.Parameter("sigma", sigma_init,  0, 10)
        low = zfit.Parameter("low", 0,  -2, 2)
        high = zfit.Parameter("high", 16, 10, 18)
        """ Define gaussian model: """
        gauss = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
        '''
        """ Define uniform background: """
        print("defining uniform")
        unif = zfit.pdf.Uniform(low, high, obs)
        """ Define the pdf sum: """
        sig_frac = zfit.Parameter("sig_frac", 0.8, 0, 1)
        print("defining sum")
        model = zfit.pdf.SumPDF([gauss, unif], [sig_frac])
        '''
    except zfit.util.exception.NameAlreadyTakenError:
        logging.warning("Fit model already defined, using pre-defined one...")

    """ Fit to residuals: """
    print("model defined")
    data = zfit.Data.from_numpy(obs=obs, array=timing_data)
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)
    #print(result.params)

    """ Calculate scale of the model to plot the fit curve later: """
    scale = len(timing_data) / timing_bins * data.data_range.area()

    return gauss, result, scale

def saturated_fun(x, B, a, b): 
    #return B*(1-np.exp(-a*x + b))
    return (a*x+b)/(1+B*(a*x+b)) 
def saturated_linear(x, m, q, k):
    return (m*x + q)/(1+k*(m*x+q))
def saturated_expo(x, B, a, b): 
    return B*(1-np.exp(-a*x + b))

def Average(lst):
    return sum(lst) / len(lst)

def get_mode(sub_group):
    return sub_group.mode().iloc[0]

def jack_knife(sample):
    std_list =[]
    num_ev = len(sample)
    pitch = math.ceil(num_ev/10)
    print(pitch)
    sample_copy = sample.copy()
    i=0
    while (len(sample_copy)>0):
        print(i)
        if ((i+1)*pitch)>(len(sample)-1):
            std_list.append(np.std(sample[i*pitch: len(sample)]))
            sample_copy = []
        else:
            std_list.append(np.std(sample[i*pitch: (i+1)*pitch]))
            sample_copy = sample_copy[(i+1)*pitch:]
        print(sample_copy)
        i += 1
        print(std_list)
    return (np.mean(std_list), np.std(std_list))
        


def get_third(sub_group):
    sub_group_copy = sub_group.copy()
    sub_group_copy = sub_group_copy.sort_values()
    if len(sub_group)-3 < 0:
        return sub_group_copy.iloc[len(sub_group)-1]
    else:
        return sub_group_copy.iloc[len(sub_group)-3]

def main():
    hep.style.use(hep.style.CMS)
    colors = ["red", "green", "black", "magenta", "pink", "yellow", "orange", "brown", "darkbrown", "violet", "cyan"]
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='Analyse data from ME0 stack testbeam july 2023',
                        epilog='Text at the bottom of help')
    parser.add_argument('filename')           # positional argument
    #parser.add_argument('-c', '--count')      # option that takes a value
    args = parser.parse_args()
    colors = ["red", "orange", "magenta", "green", "cyan", "black", "blue"]
    # Create a new colormap with the modified colors
    custom_cmap = ListedColormap(base_colors)
    #print(args.filename, args.count, args.verbose)
    background = pd.read_csv("/eos/home-f/fnenna/background_data_on.csv")
    run_file = args.filename   #ex. "00000342.root"
    root_fileDigi = "/eos/user/f/fnenna/" + run_file + "-digi.root"
    fileDigi = uproot.open(root_fileDigi)
    print(fileDigi.classnames())
    t_digi = fileDigi["outputtree"]
    t_digi.show()
    branches_digi = t_digi.arrays(entry_stop = entry_stop1)
    digiStrip = branches_digi["digiStrip"]
    digiTime = branches_digi["digiStripTime"]
    digiChamber = branches_digi["digiStripChamber"]
    digiEta = branches_digi["digiStripEta"]
    print(digiStrip)
    print(digiTime)
    dic={"strips":[], "cluster_size":[], "cluster_time":[], "event":[]}
    fig_digi, ax_digi = plt.subplots(2,3, figsize = (30, 20))
    ax_digi = ax_digi.flatten()
    fig_rec, ax_rec = plt.subplots(2,3, figsize = (30, 20))
    ax_rec = ax_rec.flatten()
    for chamber in range(0, 6):   #loop on the chambers
        for eta in range(1, 9):   #loop on the eta partitions
            cut_digi = (digiChamber == chamber) & (digiEta == eta)
            strips_sel = digiStrip[cut_digi][ak.num(digiStrip[cut_digi], axis = 1)>0]
            digi_time_sel = digiTime[cut_digi][ak.num(digiTime[cut_digi], axis = 1)>0]
            #print(strips_sel)
            #print(len(strips_sel))
            event_id = np.arange(0, len(strips_sel))    
            event_id_broad, _ = ak.broadcast_arrays(event_id, strips_sel)   #associate an event ID to the digis and digi_time
            #print(event_id_broad)
            for event in np.arange(0, len(strips_sel)):     #loop on the number of events
                cut_event = ak.firsts(event_id_broad)==event
                strip_ev = ak.flatten(strips_sel[cut_event])
                time_ev = ak.flatten(digi_time_sel[cut_event])
                #At this point I have the strips fired and the time associated
                #I want to order the time information with respect to 
                #the strip number it belong to
                #couple the strip number to the timing information
                combined = list(zip(strip_ev, time_ev))
                # Step 2: Sort the combined list based on the reference list, i.e. the strip number
                sorted_combined = sorted(combined, key=lambda x: x[0])
                # Step 3: Separate the lists
                sorted_strip, sorted_time = zip(*sorted_combined)
                # Convert back to lists
                sorted_strip = list(sorted_strip)
                sorted_time = list(sorted_time)
                #print(sorted_strip)
                cluster =[]
                cluster_time =[]
                #initialize the lists and add the first strip
                cluster.append(sorted_strip[0])
                cluster_time.append(sorted_time[0])
                renew = False
                print(f"cluster: {cluster}")
                if len(sorted_strip)==1:    #if only one strip is fired
                    dic["strips"].append(cluster)
                    dic["cluster_size"].append([1])
                    dic["cluster_time"].append(cluster_time)
                    dic["event"].append(event)
                else:   #if more than one strip is fired
                    for index in range(len(sorted_strip)-1):    #loop on the number of strips fired
                        #print(index)
                        if renew:   #use a boolean variable needed to append the first term of a new cluster if the previous is completed
                            cluster.append(sorted_strip[index])
                            cluster_time.append(sorted_time[index])
                        renew = False
                        if abs(sorted_strip[index] - sorted_strip[index+1]) == 1:   #two adjacent strips are fired
                            cluster.append(sorted_strip[index+1])
                            cluster_time.append(sorted_time[index+1])
                        else:   #otherwise, the successive hit does not belong to the cluster, so I fill the dictionary with the cls size, strips, time and evt
                            cluster_size = list(len(cluster)*np.ones(len(cluster)))
                            cls_event = list(event*np.ones(len(cluster)))
                            print(f"cluster: {cluster}")
                            dic["strips"].append(cluster)
                            dic["cluster_size"].append(cluster_size)
                            dic["cluster_time"].append(cluster_time)
                            dic["event"].append(event)
                            renew = True    #renew back to true so that the first term of the next cluster is considered
                            cluster =[]
                            cluster_size=[]
                            cluster_time=[]
                            cls_event =[]
                    #at the end of the list close the last cluster
                    cluster_size = list(len(cluster)*np.ones(len(cluster)))
                    cls_event = list(event*np.ones(len(cluster)))
                    dic["strips"].append(cluster)
                    dic["cluster_size"].append(cluster_size)
                    dic["cluster_time"].append(cluster_time)
                    dic["event"].append(event)
        df = pd.DataFrame(dic)
        print(df)
        #############################################################################################
        #################################TIMING######################################################
        print(strips_sel)
        print(event_id_broad)
        #df["strips"] = np.round(ak.mean(df["strips"], axis = 1))
        df["central_strip"] = ak.mean(df["strips"], axis = 1)
        #df["strips"] = df["strips"] - df["central_strip"]
        df['strips'] = df.apply(lambda row: [row['central_strip'] - x for x in row['strips']], axis=1)
        df["cluster_size"] = df["cluster_size"].str[0]
        df = df[df["cluster_size"]>1]
        print(df)
        print(ak.flatten(df["strips"]))
        print( ak.flatten(df["cluster_time"]))
        new_df_even = pd.DataFrame()
        new_df_odd = pd.DataFrame()
        new_df_even["strips"] = ak.flatten(df["strips"][df["cluster_size"]%2 == 0])
        new_df_even["cluster_time"] = ak.flatten(df["cluster_time"][df["cluster_size"]%2 == 0])
        new_df_odd["strips"] = ak.flatten(df["strips"][df["cluster_size"]%2 == 1])
        new_df_odd["cluster_time"] = ak.flatten(df["cluster_time"][df["cluster_size"]%2 == 1])
        print(new_df_odd)
        print(new_df_even)
        #strips
        cls_size_even = np.arange(min(new_df_even["strips"]), max(new_df_even["strips"]))
        print(cls_size_even)
        cls_size_odd = np.arange(min(new_df_odd["strips"]), max(new_df_odd["strips"]))
        print(cls_size_odd)
        avg_time_even = []
        std_time_even = []
        avg_time_odd = []
        std_time_odd = []
        for i in range(len(cls_size_even)):
        #for i in [1]:
            print(i)
            print(cls_size_even[i])
            print(new_df_even["strips"][new_df_even["strips"]==cls_size_even[i]])
            print(new_df_even["cluster_time"][new_df_even["strips"]==cls_size_even[i]])
            avg_time_even.append(np.mean(new_df_even["cluster_time"][new_df_even["strips"]==cls_size_even[i]].to_numpy()))
            std_time_even.append(np.std(new_df_even["cluster_time"][new_df_even["strips"]==cls_size_even[i]].to_numpy())/len(new_df_even["cluster_time"][new_df_even["strips"]==cls_size_even[i]].to_numpy()))
        for i in range(len(cls_size_odd)):
        #for i in [15]:
            print(new_df_odd["strips"][new_df_odd["strips"]==cls_size_odd[i]])
            print(new_df_odd["cluster_time"][new_df_odd["strips"]==cls_size_odd[i]])
            avg_time_odd.append(np.mean(new_df_odd["cluster_time"][new_df_odd["strips"]==cls_size_odd[i]].to_numpy()))
            std_time_odd.append(np.std(new_df_odd["cluster_time"][new_df_odd["strips"]==cls_size_odd[i]].to_numpy())/len(new_df_odd["cluster_time"][new_df_odd["strips"]==cls_size_odd[i]].to_numpy()))
        print(avg_time_odd)
        fig, ax = plt.subplots(1, figsize = (10, 10))
        ax.errorbar(cls_size_even, avg_time_even, yerr = std_time_even, xerr = 0.5,color = "blue",  marker = "s", label = "even cls size")
        ax.errorbar(cls_size_odd, avg_time_odd, yerr = std_time_odd, xerr = 0.5,color = "red", marker = "s", label = "odd cls size")
        ax.set_xlabel("relative strip")
        ax.set_ylabel("avg arrival time")
        #ax.set_xlim(-10, 10)
        ax.legend()
        fig.suptitle(f"chamber {chamber}")
        fig.tight_layout()
        fig.savefig(f"{run_number}/timing_cluster_strips_chamber{chamber}(2).png")
        fig2, ax2 = plt.subplots(1, figsize = (10, 10))
        hist = ax2.hist2d(new_df_odd["strips"], new_df_odd["cluster_time"], bins = (39, 7), range = ((-19.5, 19.5),(0.5,7.5)), cmap = custom_cmap)
        plt.colorbar(hist[3], label='Counts')
        ax2.set_xlabel("relative strip")
        ax2.set_ylabel("avg arrival time")
        fig2.suptitle(f"chamber {chamber} - odd cls size")
        fig2.tight_layout()
        fig2.savefig(f"{run_number}/timing_cluster_strips_chamber{chamber}_odd_hist2d.png")
        plt.close(fig2)
        fig2, ax2 = plt.subplots(1, figsize = (10, 10))
        hist = ax2.hist2d(new_df_even["strips"], new_df_even["cluster_time"], bins = (39, 7), range = ((-19.5, 19.5),(0.5,7.5)), cmap = custom_cmap)
        figs, axs = plt.subplots(1, 2, figsize = (20, 10))
        j = 0
        for k in [-3, -2, -1, 0, 1, 2, 3]:
            axs[0].hist(new_df_odd["cluster_time"][new_df_odd["strips"]==k], bins = 7, range = (0.5, 7.5), label = f"strip {k}", color = colors[j], histtype = "step", linewidth = 3)
            axs[0].legend()
            axs[0].set_xlabel("arrival time (BX)")
            axs[0].set_ylabel("digis / BX")
            axs[0].set_title(f"chamber {chamber} - odd")
            j+=1
        j = 0
        for k in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            axs[1].hist(new_df_even["cluster_time"][new_df_even["strips"]==k], stacked = True, bins = 7, range = (0.5, 7.5), label = f"strip {k}", color = colors[j], histtype = "step", linewidth = 3)
            axs[1].legend()
            axs[1].set_xlabel("arrival time (BX)")
            axs[1].set_ylabel("digis / BX")
            axs[1].set_title(f"chamber {chamber} - even")
            j+=1
        figs.tight_layout()
        figs.savefig(f"{run_number}/timing_cluster_strips_chamber{chamber}_hist1d.png")
        plt.colorbar(hist[3], label='Counts')
        ax2.set_xlabel("relative strip")
        ax2.set_ylabel("avg arrival time")
        fig2.suptitle(f"chamber {chamber} - even cls size")
        fig2.tight_layout()
        fig2.savefig(f"{run_number}/timing_cluster_strips_chamber{chamber}_even_hist2d.png")
        #print(digi_time_sel)

if __name__ == "__main__":
    main()
