---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import numpy as np
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os
import ntpath
```

```python
import pandas as pd
pd.options.display.max_rows = 150
```

```python
def save_fig(filename, folder, trans=True, dpi=800, bbox_inches='tight', pad=0.1):
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig('{}/{}.png'.format(folder, filename), transparent=trans, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad) 
    return 
```

```python
def rebin(data, rebin_ratio, do_average=False):
    """ Rebin a 1D array the good old way.
    @param 1d numpy array data : The data to rebin
    @param int rebin_ratio: The number of old bin per new bin
    @return 1d numpy array : The array rebinned
    The last values may be dropped if the sizes do not match. """
    
    rebin_ratio = int(rebin_ratio)
    length = (len(data) // rebin_ratio) * rebin_ratio
    data = data[0:length]
    data = data.reshape(length//rebin_ratio, rebin_ratio)
    if do_average :
        data_rebinned = data.mean(axis=1)
    else :
        data_rebinned = data.sum(axis=1)
    return data_rebinned

def decimate(data, decimation_ratio):
    """ Decimate a 1D array . This means some value are dropped, not averaged
    @param 1d numpy array data : The data to decimated
    @param int decimation_ratio: The number of old value per new value
    @return 1d numpy array : The array decimated. """
    
    decimation_ratio = int(decimation_ratio)
    length = (len(data) // decimation_ratio) * decimation_ratio
    data_decimated = data[:length:decimation_ratio]
    return data_decimated

def rebin_xy(x, y, ratio=1, do_average=True):
    """ Helper method to decimate x and rebin y, with do_average True as default. """
    return decimate(x, ratio), rebin(y, ratio, do_average)
```

# Data

```python
# Specify the main directory where your data folders are located
main_directory = r'C:\Users\YB274940\Desktop\pl_spectrum_V_samples\20240426'

# Get a list of all items in the main directory
items = os.listdir(main_directory)

# Create an empty list to store DataFrames
dfs = []

# Loop through each item in the main directory
for item in items:
    item_path = os.path.join(main_directory, item)
    
    # Check if the item is a file
    if os.path.isfile(item_path) and item.endswith('.txt'):
        # Read the txt file into a DataFrame
        df = pd.read_csv(item_path, sep='\t', skiprows=0)
        # Append the DataFrame to the list of DataFrames
        dfs.append(pd.Series({'wavelength': np.array(df.iloc[:,0]), 'pl_intensity': np.array(df.iloc[:,1]), 'filename': ntpath.basename(item)}).to_frame().transpose())
    
    # Check if the item is a directory
    elif os.path.isdir(item_path):
        # Loop through each file in the subdirectory
        for file in os.listdir(item_path):
            if file.endswith('.txt'):
                file_path = os.path.join(item_path, file)
                # Read the txt file into a DataFrame
                df = pd.read_csv(file_path, sep='\t', skiprows=0)
                # Append the DataFrame to the list of DataFrames
                dfs.append(pd.Series({'wavelength': np.array(df.iloc[:,0]), 'pl_intensity': np.array(df.iloc[:,1]), 'filename': ntpath.basename(file)}).to_frame().transpose())

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(dfs, ignore_index=True)
```

```python
final_df
```

```python
# Spectrometer offset on the grating at 75 lines
offset_spectro = 14
```

# Plot data

```python
fig, ax = plt.subplots(figsize=(8, 4))

j=0
for i, row in final_df.iterrows():
    ax.plot(row.wavelength, row.pl_intensity+j*1e3, marker='None', label=row.filename)
#     , label=row.filename[9:-22])
    j += 1
    
    
ax.axvline(1218, color='red', ls='--', alpha=0.5, label='W-line')
ax.axvline(1419, color='orange', ls='--', alpha=0.5, label='D2-line')
ax.axvline(1536, color='orange', ls='--', alpha=0.5, label='D1-line')
ax.axvline(1450, color='forestgreen', ls='--', alpha=0.5, label='1450-line')
ax.set_ylim(0, None)
# ax.legend(ncol=3)
pass
```

# Subdata

```python
# Assuming you've already filtered the DataFrame as shown in the previous example
given_string = 'V1D'
filtered_df = final_df[final_df['filename'].str.contains(given_string)]

# Iterate through the rows and print information
for index, row in filtered_df.iterrows():
    filename = row['filename']
    wavelength_length = len(row['wavelength'])

    print(f"Filename: {filename}, Wavelength Length: {wavelength_length}")
```

# Plot subdata

```python
fig, ax = plt.subplots(figsize=(8, 4))

j=0
color=['dodgerblue', 'forestgreen', 'crimson', 'indigo']
for i, row in filtered_df.iterrows():
    if "filter" not in row['filename'] and "03" not in row['filename']:
        ax.plot(row.wavelength, row.pl_intensity, marker='None', label=row.filename, color=color[j])
        ax.fill_between(row.wavelength, row.pl_intensity, lw=0, color=color[j], alpha=0.25)
#         ax.axhline(j*0.15e3, color='black', ls='-', alpha=0.1)
        j += 1
    
ax.set_xlim(950, 1670)
# ax.set_yscale('log')
ax.set_ylim(-0,None)
# ax.fill_between([950, 1200], 0.15e3, 0.30e3, color='gray', alpha=0.1)
# ax.fill_between([1300, 1600], 0.15e3, 0.30e3, color='gray', alpha=0.1)

ax.axvline(1218, color='red', ls='--', alpha=0.5, label='W-line')
# ax.axvline(1232, color='red', ls='--', alpha=0.1)

ax.axvline(1419, color='orange', ls='--', alpha=0.5, label='D2-line')
ax.axvline(1536, color='black', ls='--', alpha=0.5, label='D1-line')
ax.axvline(1450, color='forestgreen', ls='--', alpha=0.5, label='1450-line')
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("PL [c/s]")
ax.legend(ncol=2)
pass
```

```python
fig, ax = plt.subplots(figsize=(8, 4))

j=0
color=['dodgerblue', 'forestgreen', 'crimson', 'indigo']
for i, row in filtered_df.iterrows():
    if "filter" not in row['filename']:
        ax.plot(row.wavelength, row.pl_intensity+j*0.15e3, marker='None', label=row.filename, color=color[j])
        j += 1
    
ax.set_xlim(950, 1500)
ax.set_ylim(None, 2e3)
# ax.fill_between([950, 1200], -50, 0.15e3, color='gray', alpha=0.1)
# ax.fill_between([1300, 1500], -50, 0.15e3, color='gray', alpha=0.1)

ax.axvline(1218, color='red', ls='--', alpha=0.5, label='W-line')
# ax.axvline(1232, color='red', ls='--', alpha=0.5, label='W-line bis')
ax.axvline(1419, color='orange', ls='--', alpha=0.5, label='D2-line')
# ax.axvline(1536, color='orange', ls='--', alpha=0.5, label='D1-line')
ax.axvline(1450, color='forestgreen', ls='--', alpha=0.5, label='1450-line')
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("PL [c/s]")
ax.legend(ncol=1)
pass
```

# Simple spectrum

```python
# filename = r"C:\Users\YB274940\Desktop\Data_run14\20230828\20230828_Q4E_0p8mW.txt" # Q4E
filename = r"C:\Users\YB274940\Desktop\20240403_T1A02_sans_filtre.txt" # S1A
# filename = r"S:\120-LETI\120.21-DPFT\120.21.14-SSURF\120.21.14.2-Info_communes\WTeq\Data_run14\20230829\20230829_T2A_Punknown.txt"

# filename = r"S:\120-LETI\120.21-DPFT\120.21.14-SSURF\120.21.14.2-Info_communes\WTeq\Data_run14\20230830\20230830_T2B_f50um.txt"
# filename = r"S:\120-LETI\120.21-DPFT\120.21.14-SSURF\120.21.14.2-Info_communes\WTeq\Data_run14\20230830\20230830_T2C_f50um.txt"

# filename = r"S:\120-LETI\120.21-DPFT\120.21.14-SSURF\120.21.14.2-Info_communes\WTeq\Data_run14\20230831\20230831_T3D.txt"
```

```python
# IMPORT DATA
data = pd.read_csv(filename, sep='	', skiprows=0)
data['Wavelength_u'] = data['Wavelength']/1e3
Wavelength = np.array(data["Wavelength"], dtype=float)
S1 = np.array(data["S1"], dtype=float)

x_min, x_max = data["Wavelength"].min()/1e3-offset_spectro, data["Wavelength"].max()/1e3-offset_spectro
y_min, y_max = 0,(data["S1"].max() + data["S1"].max()*20/100)/1e3

# PLOT
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("Wavelength [$\mu m$]")
ax.set_ylabel("PL [kc/s]")
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.plot(data["Wavelength_u"]-offset_spectro, data["S1"]/1e3, marker='None', c="dodgerblue")
ax.fill_between(Wavelength/1e3-offset_spectro, S1/1e3, color='dodgerblue', alpha=0.2)
ax.axvline(x=1.218, color='black', ls='--', lw=0.75, alpha=0.4)

# ax.fill_between([0.94,1.2], 2, color='crimson', alpha=0.05)
# ax.text(0.98, y_max-y_max*0.1, 'Filter', color='darkred', fontweight='bold', alpha=0.7)

# ax.text(0.98, y_max-y_max*0.1, 'Q4E sample', color='forestgreen', fontweight='bold')
# ax.text(0.98, y_max-y_max*0.1, 'S1A sample', color='forestgreen', fontweight='bold')
# ax.text(0.98, y_max-y_max*0.1, 'T2B sample', color='forestgreen', fontweight='bold')
# ax.text(0.98, y_max-y_max*0.1, 'T2C sample', color='forestgreen', fontweight='bold')
# ax.text(0.98, y_max-y_max*0.1, 'T2A sample', color='forestgreen', fontweight='bold')
# ax.text(1.24, y_max-y_max*0.1, 'T3D sample', color='forestgreen', fontweight='bold')
pass
```

# Double spectra

```python
# filename = r"C:\Users\YB274940\Desktop\Data_run14\20230828\20230828_T1A_0p8mW.txt" # T1A (28/08/2023)
# filename_zoom = r"C:\Users\YB274940\Desktop\Data_run14\20230828\20230828_T1A_0p8mW_900tr.txt" # T1A (28/08/2023)

filename = r"S:\120-LETI\120.21-DPFT\120.21.14-SSURF\120.21.14.2-Info_communes\WTeq\Data_run14\20230830\20230830_T2D_f50um.txt"
filename_zoom = r"S:\120-LETI\120.21-DPFT\120.21.14-SSURF\120.21.14.2-Info_communes\WTeq\Data_run14\20230830\20230830_T2D_f50um_900tr.txt"
```

```python
data = pd.read_csv(filename, sep='	', skiprows=0)
data_zoom = pd.read_csv(filename_zoom, sep='	', skiprows=0)

data['Wavelength_u'] = data['Wavelength']/1e3
Wavelength = np.array(data["Wavelength"], dtype=float)
S1 = np.array(data["S1"], dtype=float)

data_zoom['Wavelength_u'] = data_zoom['Wavelength']/1e3
Wavelength_zoom = np.array(data_zoom["Wavelength"], dtype=float)
S1_zoom = np.array(data_zoom["S1"], dtype=float)

# PLOT (75 lines)
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(data["Wavelength"].min()-offset_spectro*1e3+120
            , data["Wavelength"].max()-offset_spectro*1e3)
ax.set_ylim(0,(data["S1"].max() + data["S1"].max()*2/100)/1e3)
ax.set_xlabel("Wavelength [$nm$]")
ax.set_ylabel("PL [kc/s]")
ax.xaxis.set_major_locator(plt.MaxNLocator(9))

ax.plot(data["Wavelength"]-offset_spectro*1e3, data["S1"]/1e3, marker='None', c="dodgerblue", label='T2D sample')
ax.fill_between(Wavelength-offset_spectro*1e3, S1/1e3, color='dodgerblue', alpha=0.2)
ax.fill_between([data_zoom["Wavelength"].min(), data_zoom["Wavelength"].max()], 10, color='crimson', alpha=0.05)
ax.axvline(x=1218, color='gray', ls='--', lw=0.75, label='W-center ZPL')

ax.text(1225, 2.5, 'W-center ZPL', color='gray', fontweight='bold')
ax.text(1140, 8.5, 'T2D sample', color='dodgerblue', fontweight='bold')

# Inset (900 lines) #

ax2 = plt.axes([0.56, 0.5, 0.32, 0.35])

ax2.set_xlim(data_zoom["Wavelength"].min(), data_zoom["Wavelength"].max())
ax2.set_ylim(0,(data_zoom["S1"].max() + data_zoom["S1"].max()*2/100))
ax2.set_xlabel("Wavelength [$nm$]")
ax2.set_ylabel("PL [c/s]")
ax2.xaxis.set_major_locator(ticker.FixedLocator([1210, 1220, 1230]))

ax2.plot(data_zoom["Wavelength"], data_zoom["S1"], marker='None', c="crimson", label='T2D sample')
ax2.fill_between(Wavelength_zoom, S1_zoom, color='crimson', alpha=0.2)
ax2.axvline(x=1218, color='gray', ls='--', lw=0.75, label='W-center ZPL')
ax2.text(1203, 300, 'Zoom on ZPL', color='crimson', fontweight='bold')
ax2.text(1202.5, 270, '(using high-res grating)', color='crimson', fontsize=7)
# ax.legend(loc=2, fontsize=10)
# ax2.legend(loc=2, fontsize=8)

folder = r'C:\Users\YB274940\Desktop'
save_fig('T2D', folder, dpi=800, bbox_inches='tight', pad=0.1)
pass
```

# Broad Spectrum

```python
filename = r'C:\Users\YB274940\Desktop\Data_run14\20230828\20230828_T1A_0p8mW_900tr_broad.txt' #T1A HD (28/08/2023)
```

```python
data = pd.read_csv(filename, sep='	', skiprows=0)
data['Wavelength_u'] = data['Wavelength']/1e3
Wavelength_u = np.array(data["Wavelength_u"], dtype=float)
S1 = np.array(data["S1"], dtype=float)
x_to_plot, y_to_plot = rebin_xy(Wavelength_u, S1, ratio=2)
x_min, x_max = data["Wavelength"].min()/1e3, data["Wavelength"].max()/1e3
# y_min, y_max = 0,(data["S1"].max() + data["S1"].max()*2/100)/1e3
y_min, y_max = 0.01,(data["S1"].max() + data["S1"].max()*20/100)/1e3

# PLOT
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("Wavelength [$\mu m$]")
ax.set_ylabel("PL [kc/s]")
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.plot(x_to_plot, y_to_plot/1e3, marker='None', c="dodgerblue")
ax.fill_between(x_to_plot, y_to_plot/1e3, color='dodgerblue', alpha=0.2)

ax.axvline(x=1.218, color='black', ls='--', lw=0.75, alpha=0.4)
# ax.text(1.0, y_max-y_max*0.4, 'T1A sample', color='forestgreen', fontweight='bold')
ax.set_yscale('log')
pass
```
