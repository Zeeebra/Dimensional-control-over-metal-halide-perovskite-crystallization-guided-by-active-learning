# select samples using Kennard-Stone algorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets, interactive

def UV_process (filename, blank = ['B01','B02','B03','B04'], window_length = 31, polyorder = 2):

    # import spectra file as panda dataframe
    UV_orig = pd.read_csv('Abs/'+filename, sep='\t')

    # transpose the dataframe so that index is vial ID and column is wavelength
    UV = UV_orig.transpose()
    columns = np.array(UV.loc['Wavelength'], dtype='int32')
    UV = pd.DataFrame(data = np.array(UV.drop(['Wavelength'])), columns = columns, index = UV.drop(['Wavelength']).index)

    # Clean up the spectra: 
    # 1) remove NA data ('NA' means no spectra were measured) 2) remove blank 3) remove short and long wavelength
    UV = UV.dropna()
    UV = UV.drop(blank)
    UV = UV.drop(np.hstack((np.arange(350,415),np.arange(801,2501))), axis=1)

    # Process the spectra
    UV_bs = UV.subtract(UV[800],axis = 0) # substract the baseline at 800 nm
    UV_norm = UV_bs.div(UV_bs.max(axis=1), axis = 0) # normalize the spectra

    # Smooth UV_norm data with Savitzky-Golay
    from scipy.signal import savgol_filter
    UV_norm_sm = pd.DataFrame(columns = UV_norm.columns)
    for i in UV_norm.index:
        UV_norm_sm.loc[i] = savgol_filter(UV_norm.loc[i], window_length = window_length, polyorder = polyorder, delta=1)

    # Calculate tauc plot
    hv = 1240/UV_norm_sm.columns
    UV_dir = (UV_norm_sm.mul(hv, axis = 1))**2
    UV_dir.columns = hv

    # Calculate 1st order derivative of UV_dir
    UV_dir_dev1 = pd.DataFrame(columns = UV_dir.columns[:-1])
    for i in UV_dir.index:
        diff_y = np.diff(np.array(UV_dir.loc[i]))
        diff_x = np.diff(np.array(UV_dir.columns))
        UV_dir_dev1.loc[i] = diff_y/diff_x

    # Calculate 2nd order derivative of UV_dir
    UV_dir_dev2 = pd.DataFrame(columns = UV_dir_dev1.columns[:-1])
    for i in UV_dir_dev1.index:
        diff_y = np.diff(np.array(UV_dir_dev1.loc[i]))
        diff_x = np.diff(np.array(UV_dir_dev1.columns))
        UV_dir_dev2.loc[i] = diff_y/diff_x

    # Smooth 2nd derivative of UV_dir data with Savitzky-Golay (for better peak fitting)
    UV_dir_dev2_sm = pd.DataFrame(columns = UV_dir_dev2.columns)
    for i in UV_dir_dev2.index:
        UV_dir_dev2_sm.loc[i] = savgol_filter(UV_dir_dev2.loc[i], window_length = window_length, polyorder = polyorder)

    # Save all processed spectra to csv files
    (UV.T).to_csv('Abs/'+filename[:-4]+'_original.csv')
    (UV_bs.T).to_csv('Abs/'+filename[:-4]+'_bs.csv')
    (UV_norm.T).to_csv('Abs/'+filename[:-4]+'_norm.csv')
    (UV_dir.T).to_csv('Abs/'+filename[:-4]+'_dir.csv')

    spec_dict = {'original': UV,
                 'basline corrected': UV_bs,
                 'normalized': UV_norm_sm,
                 'tauc plot (direct)': UV_dir, 
                 '1st derivative of tauc plot': UV_dir_dev1,
                 '2nd derivative of tauc plot': UV_dir_dev2_sm}
    
    return spec_dict


def UV_plot (spec_dict, legend_size = 10):

    # Make a dropdown to select well ID and spectral type
    wellid = widgets.Dropdown(options = ['All'] + list(spec_dict['original'].index), value = 'All', description = 'Well ID')
    spectype = widgets.Dropdown(options = list(spec_dict.keys()),\
                                value = list(spec_dict.keys())[0], description = 'Spectral Type')
    # Plot absorption spectra
    def specplot (wellid, spectype):
        df_plot = spec_dict[spectype]
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot()

        if wellid == 'All':
            for idx in df_plot.index:
                ax.plot(df_plot.columns, df_plot.loc[idx], label = idx)
                ax.legend(loc='best', prop={'size': legend_size})
        else:
            df_plot = df_plot.loc[wellid]
            ax.plot(list(df_plot.index), list(df_plot), label = wellid)
            ax.legend(loc='best', prop={'size': legend_size})

    return interactive(specplot, wellid = wellid, spectype = spectype)


def Tauclinfit (tauc, tauc_dev2, filename, prominence = 10):

    from scipy.signal import argrelextrema, find_peaks

    tauc_NA = []
    tauc_linear_range = dict() # the index of the linear range in the tauc plot
    for i in tauc_dev2.index:
        peaks, _ = find_peaks(np.array(tauc_dev2.loc[i]), prominence=prominence)
        valeys,_ = find_peaks(-np.array(tauc_dev2.loc[i]), prominence=prominence)
        if (peaks.size != 0) & (valeys.size != 0):
            if peaks[-1] > valeys[-1]:
                tauc_linear_range[i] = (peaks[-1], valeys[-1])
            else:
                tauc_NA.append(i)
        else:
            tauc_NA.append(i)

    print ("The current setting of parameters can't fit", tauc_NA, 'so you have to do them manually')

    tauc_linear = dict() # the linear range in the tauc plot
    for i,j in tauc_linear_range.items():
        tauc_linear[i] = (tauc.filter([i],axis=0)).iloc[0,j[1]:j[0]+1]
        tauc_linear[i] = pd.DataFrame(tauc_linear[i]).T

    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot()

    # dataframe of bandgap values and R2 of the linear fittings
    tauc_bandgap = pd.DataFrame(columns = ['bandgap','R2'], index = list(tauc_linear.keys()))


    for vial in tauc_linear:
        # fit x,y with linear regression model
        LR = linear_model.LinearRegression()
        LR.fit(np.array(tauc_linear[vial].columns).reshape(-1,1), np.array(tauc_linear[vial].iloc[0])) 

        # generate x,y for plotting the linear fitting
        x_linear = np.linspace(2,3,200).reshape(-1,1)
        y_linear = LR.predict(x_linear)

        # calculate and record bandgap and R2 from the linear fitting
        tauc_bandgap['bandgap'].loc[vial] = [-(LR.intercept_)/(LR.coef_)][0][0]
        tauc_bandgap['R2'].loc[vial] = r2_score(np.array(tauc_linear[vial].iloc[0]), LR.predict(np.array(tauc_linear[vial].columns).reshape(-1,1)))

        # plot tauc plot and linear fitting.
        tauc_plot = tauc.filter([vial], axis = 0)
        for idx in tauc_plot.index:
            ax.plot(tauc_plot.columns, tauc_plot.loc[idx], label = idx)
            ax.legend(loc='best', prop={'size': 7})
            ax.plot(x_linear,y_linear, c = "red", linestyle='dashed', linewidth=0.5)


    ax.set_xlim(1.9,2.9)
    ax.set_ylim(-0.5, 8)
    ax.xaxis.set_ticks(np.arange(2.0,2.9,0.1))
    ax.set_xlabel('hv (eV)')
    ax.set_ylabel('(Ahv)2 (a.u.)')
    plt.show()
    plt.savefig('Abs/'+ filename[:-4] + '_tauc_linearft.svg', format = "svg", transparent=True)

    # save bandgap data and R2 values in a csv file
    tauc_bandgap.sort_values(axis = 0, by = 'bandgap', ascending = False, inplace = True)
    tauc_bandgap.to_csv('Abs/'+filename[:-4]+'_bandgap.csv')