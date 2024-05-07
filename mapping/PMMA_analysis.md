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
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, ConstantModel
from pathlib import Path
from matplotlib.colors import LogNorm
import data_analysis_toolbox as dat
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import pandas as pd
pd.options.display.max_rows = 150
pd.options.display.max_columns = 150
```

```python
from ast import literal_eval
# Function to convert string representation of tuple to tuple
def str_to_tuple(s):
    try:
        return literal_eval(s)
    except (ValueError, SyntaxError) as e:
        return None  # Return None if conversion fails
```

# DATA

```python
folder = r"C:\Users\YB274940\qudi\Data\2024\03\2024-03-28"
data = dat.get_dataframe_from_folders(folder)
```

# PLOT RAW MAP

```python
def plot_map(ax, scan, extent, norm=None, xlim=None, ylim=None):
    divider = make_axes_locatable(ax)
    ax_bar = divider.append_axes('right', size=.2, pad=.05)
    
    im = ax.imshow(scan, cmap='inferno', extent=extent, aspect='equal', norm=norm)
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
        
    if norm:
        cbar = fig.colorbar(im, cax=ax_bar, label='PL intensity [kcps]', orientation='vertical', norm=norm)
    else:
        cbar = fig.colorbar(im, cax=ax_bar, label='PL intensity [kcps]', orientation='vertical')
    ax_bar.yaxis.tick_right()
    ax_bar.yaxis.set_label_position('right')
    cbar.ax.tick_params(direction='in', which='both')
```

```python
scan = data["_raw_data"][0][::-1] / 1e3
xmin, xmax, ymin, ymax = -100, 100, -60, 60
extent = [xmin, xmax, ymin, ymax]
plot_settings = {"dpi": 100, "figsize": (5, 5)}

# Linear scale
fig, ax = plt.subplots(**plot_settings)
plot_map(ax, scan, extent)
# Log scale
fig, ax = plt.subplots(**plot_settings)
plot_map(ax, scan, extent, norm=LogNorm(vmin=1, vmax=1e3), xlim=(-50, 80), ylim=(-50, 60))
```

# SELECT SUBDATA (COLUMNS)

```python
column_index = 211
i_range = 15
scan_zoom_col = scan[:, column_index - i_range:column_index + i_range]

x_values = np.linspace(xmin, xmax, 601)
y_values = np.linspace(ymax, ymin, 361)
extent_col = [x_values[column_index - i_range], x_values[column_index + i_range], ymin, ymax]
```

```python
fig, ax = plt.subplots(**plot_settings)
plot_map(ax, scan, extent, norm=LogNorm(vmin=1, vmax=1e3), xlim=(-50, 80), ylim=(-50, 60))
ax.axvline(x=x_values[column_index-i_range], color='white', ls='--', lw=.8)
ax.axvline(x=x_values[column_index+i_range], color='white', ls='--', lw=.8)

fig, ax = plt.subplots(**plot_settings)
# plot_zoomed_column(ax, scan, x_values, y_values, column_index, i_range, ymin, ymax)
plot_map(ax, scan_zoom_col, extent_col, norm=LogNorm(vmin=1, vmax=1e2), ylim=(-42, 60))
ax.set_xticks([])
pass
```

# FIT COLUMN LINECUT WITH GAUSSIAN PEAKS

```python
# CALCULATE COL LINECUT
scan_per_col = [scan[:, i] for i in range(column_index - i_range, column_index + i_range)]
scan_col_avg = np.sum(scan_per_col, axis=0)

y_min, y_max = -42, 60
y_values_select, scan_col_avg_select = dat._select_data(y_values, scan_col_avg, y_min, y_max)
y_values_fit = np.linspace(y_values_select[0], y_values_select[-1], 1000)

df = pd.DataFrame({'y_values_select': y_values_select, 'peaks': scan_col_avg_select})
```

```python
#DEFINE MODEL
N = 10
model = ConstantModel(prefix='bg_') + GaussianModel(prefix='p1_')
for i in range(2, N+1):
    model += GaussianModel(prefix='p'+str(i)+'_')

#SET INITIAL VALUES
centers = [-38, -28, -16, -6, 4, 14, 24, 34, 44, 54]
amplitudes = [70, 20, 200, 20, 40, 20, 40, 20, 40, 400]
params = model.make_params(c=15)
for i, (center, amplitude) in enumerate(zip(centers, amplitudes), start=1):
    params.add(f'p{i}_center', value=center, min=-np.inf, max=np.inf)
    params.add(f'p{i}_amplitude', value=amplitude, min=0)
    params.add(f'p{i}_sigma', value=1, min=0)

#SET BOUNDS
params['bg_c'].min, params['bg_c'].max = 10, 20
for i in range(N):
    params[f'p{i+1}_center'].min, params[f'p{i+1}_center'].max = y_min, y_max
    params[f'p{i+1}_amplitude'].min = 15
    params[f'p{i+1}_sigma'].min, params[f'p{i+1}_sigma'].max = 0.5, 2
```

```python
result = model.fit(df['peaks'], params, x=df.y_values_select)
# print(result.fit_report())
```

```python
fig, ax = plt.subplots( dpi=100, figsize=(6, 4) )
ax.tick_params(direction='in', which='both')
ax.plot(df.y_values_select, df['peaks'], marker='+', markersize=4, ls='-', color='k', label='data')
ax.plot(df.y_values_select, result.best_fit, color='mediumvioletred', label='fit')
plt.legend()
ax.set_xlabel('y (µm)')
ax.set_ylabel('Line-integrated intensity (a.u.)')
pass
```

```python
# EXTRACT FITTED PARAMETERS

offset = result.params['bg_c'].value
offset_err = result.params['bg_c'].stderr

param_names = ['p'+str(i+1) for i in range(N)]
amplitude_fit = np.array([result.params[name+'_amplitude'].value for name in param_names])
amplitude_fit_err = np.array([result.params[name+'_amplitude'].stderr for name in param_names])
center_fit = np.array([result.params[name+'_center'].value for name in param_names])
center_fit_err = np.array([result.params[name+'_center'].stderr for name in param_names])
sigma_fit = np.array([result.params[name+'_sigma'].value for name in param_names])
sigma_fit_err = np.array([result.params[name+'_sigma'].stderr for name in param_names])
```

```python
#PLOT AMPLITUDE VS PEAK INDEX
index_values = np.arange(1, 11, 1)

fig, ax = plt.subplots( dpi=100, figsize=(6, 4) )
ax.tick_params(direction='in', which='both')
ax.errorbar(index_values, amplitude_fit, yerr=amplitude_fit_err, fmt='.', color='k', ecolor='k', elinewidth=.5, capthick=.5, capsize=2)
ax.set_xlabel('Peak index')
ax.set_ylabel('Amplitude (a.u.)')
pass
```

```python
#AVERAGE INTEGRATED INTENSITY FOR SINGLE EMITTERS
integrated_intensity_values = []
for i in [1, 3, 5, 7]:
    integrated_intensity_values.append(amplitude_fit[i])
integrated_intensity = np.mean(integrated_intensity_values)
integrated_intensity_err = np.std(integrated_intensity_values)
print(integrated_intensity)
print(integrated_intensity_err)
```

```python
# NORMALIZED DATA
peak_top_values = []
for i in [1, 3, 5, 7]:
    peak_top = amplitude_fit[i] / sigma_fit[i] / np.sqrt(2*np.pi)
    print(peak_top)
    peak_top_values.append(peak_top)
norm_factor = np.mean(peak_top_values)
print(norm_factor)

data_norm = ( df['peaks'] - offset ) / norm_factor
data_fit =  ( result.best_fit - offset ) / norm_factor
```

```python
#PLOT
fig, ax = plt.subplots( dpi=100, figsize=(6, 4) )
ax.tick_params(direction='in', which='both')
for i in range(13):
    ax.axhline(y=i, color='r', ls=':', lw=.8)
ax.plot(df.y_values_select, data_norm, marker='+', markersize=4, ls='-', color='k', label='data')
ax.plot(df.y_values_select, data_fit, color='r', label='fit')
ax.set_xlabel('y (µm)')
ax.set_ylabel('PL intensity norm.')
ax.set_yticks(range(13))
pass
```

# All columns

```python
df_col = []
for col_num in range(number_of_col):
    column_index = 211 + 30 * col_num
    i_range = 15
    scan_zoom_col = scan[:, column_index - i_range:column_index + i_range]
    x_values = np.linspace(xmin, xmax, 601)
    y_values = np.linspace(ymax, ymin, 361)
    extent_col = [x_values[column_index - i_range], x_values[column_index + i_range], ymin, ymax]

    # CALCULATE COL LINECUT
    scan_per_col = [scan[:, i] for i in range(column_index - i_range, column_index + i_range)]
    scan_col_avg = np.sum(scan_per_col, axis=0)
    y_min, y_max = -42, 60
    y_values_select, scan_col_avg_select = dat._select_data(y_values, scan_col_avg, y_min, y_max)
    y_values_fit = np.linspace(y_values_select[0], y_values_select[-1], 1000)

    # Create DataFrame for current column's data
    df_temp = pd.Series({'col_num': col_num+1, 'y_values_select': y_values_select, 'peaks': scan_col_avg_select}).to_frame().transpose()

    # Append the DataFrame to the list
    df_col.append(df_temp)

df_intensity_by_col = pd.concat(df_col, ignore_index=True)

for i, row in df_intensity_by_col.iterrows():
    fig, ax = plt.subplots( dpi=100, figsize=(6, 4) )
    ax.tick_params(direction='in', which='both')
    ax.plot(row.y_values_select, row['peaks'], marker='+', markersize=4, ls='-', color='k')
    ax.set_xlabel('y (µm)')
    ax.set_ylabel('Line-integrated intensity (a.u.)')
    pass
```

```python
#DEFINE MODEL
N = 10
model = ConstantModel(prefix='bg_') + GaussianModel(prefix='p1_')
for i in range(2, N+1):
    model += GaussianModel(prefix='p'+str(i)+'_')
    
centers_list = [[-38, -28, -16, -6, 4, 14, 24, 34, 44, 54],
                [-38, -28, -16, -6, 4, 14, 24, 34, 44, 54],
                [-38, -28, -16, -6, 4, 14, 24, 34, 44, 54],
                [-38, -28, -16, -6, 4, 14, 24, 34, 44, 54],
                [-38, -28, -16, -6, 4, 14, 24, 34, 44, 54],
                [-38, -28, -16, -6, 4, 14, 24, 34, 44, 54],
                [-38, -28, -16, -6, 4, 14, 24, 34, 44, 54],
                [-38, -28, -16, -6, 4, 14, 24, 34, 44, 54],
                [-38, -28, -16, -6, 4, 14, 24, 34, 44, 54],
                [-38, -28, -16, -6, 4, 14, 24, 34, 44, 54],
                [-38, -28, -16, -6, 4, 14, 24, 34, 44, 54],
                [-38, -28, -16, -6, 4, 14, 24, 34, 44, 54]
               ]
amplitudes_list = [[70, 20, 200, 20, 40, 20, 40, 20, 40, 400],
                  [70, 20, 200, 20, 40, 20, 40, 20, 40, 400],
                  [70, 20, 200, 20, 40, 20, 40, 20, 40, 400],
                  [70, 20, 200, 20, 40, 20, 40, 20, 40, 400],
                  [70, 20, 200, 20, 40, 20, 40, 20, 40, 400],
                  [70, 20, 200, 20, 40, 20, 40, 20, 40, 400],
                  [70, 20, 200, 20, 40, 20, 40, 20, 40, 400],
                  [70, 20, 200, 20, 40, 20, 40, 20, 40, 400],
                  [70, 20, 200, 20, 40, 20, 40, 20, 40, 400],
                  [70, 20, 200, 20, 40, 20, 40, 20, 40, 400],
                  [70, 20, 200, 20, 40, 20, 40, 20, 40, 400],
                  [70, 20, 200, 20, 40, 20, 40, 20, 40, 400]]

params_list = []

for centers, amplitudes in zip(centers_list, amplitudes_list):
    params = model.make_params(c=15)
    for i, (center, amplitude) in enumerate(zip(centers, amplitudes), start=1):
        params.add(f'p{i}_center', value=center, min=-np.inf, max=np.inf)
        params.add(f'p{i}_amplitude', value=amplitude, min=0)
        params.add(f'p{i}_sigma', value=1, min=0)
        
    #SET BOUNDS
    params['bg_c'].min, params['bg_c'].max = 10, 20
    for i in range(N):
        params[f'p{i+1}_center'].min, params[f'p{i+1}_center'].max = y_min, y_max
        params[f'p{i+1}_amplitude'].min = 15
        params[f'p{i+1}_sigma'].min, params[f'p{i+1}_sigma'].max = 0.5, 2    
        
    params_list.append(params)
```

```python
fit_results = []

for i, row in df_intensity_by_col.iterrows():
    result = model.fit(row.peaks, params_list[i], x=row.y_values_select)
    fit_results.append(result)
```

```python
for i, row in df_intensity_by_col.iterrows():
    fig, ax = plt.subplots( dpi=100, figsize=(6, 4) )
    ax.tick_params(direction='in', which='both')
    ax.plot(row.y_values_select, row.peaks, marker='+', markersize=4, ls='-', color='k', label='data')
    ax.plot(row.y_values_select, fit_results[i].best_fit, color='mediumvioletred', label='fit')
    plt.legend()
    ax.set_xlabel('y (µm)')
    ax.set_ylabel('Line-integrated intensity (a.u.)')
    pass
```

```python
dfs = []
number_of_col = 11
for i in range(number_of_col):
    dfs.append(pd.Series({'column_number': i+1, 'amplitude_fit': np.zeros_like(amplitude_fit)}).to_frame().transpose())

final_df = pd.concat(dfs, ignore_index=True)
```

```python
column_number = 0
final_df.at[column_number, 'amplitude_fit'] = amplitude_fit
```

```python
final_df
# .amplitude_fit[column_number]
```

# SELECT SUBDATA (LINES)

```python
# SELECT LINE
line_index = 259
i_range = 14
print(y_values[line_index])

scan_zoom_lin = scan[line_index-i_range:line_index+i_range,:]

# CALCULATE LINE LINECUT
scan_per_lin = []
for i in range(line_index-i_range, line_index+i_range):
    scan_lin = scan[i]
    scan_per_lin.append(scan_lin)

scan_per_lin = np.array(scan_per_lin)
scan_lin_avg = np.sum(scan_per_lin, axis=0)

# PLOT ON RAW MAP SELECTED LINE
fig, ax = plt.subplots( dpi=100, figsize=(5, 5) )
divider = make_axes_locatable(ax)
ax_bar = divider.append_axes('right',size=.2, pad=.05)
im = ax.imshow(scan, cmap='inferno', extent=[xmin, xmax, ymin, ymax], aspect='equal', norm=mpl.colors.LogNorm(vmin=1e0, vmax=1e3))

cbar = fig.colorbar(im, cax=ax_bar, label='PL intensity [kcps]', orientation='vertical')
ax_bar.yaxis.tick_right()
ax_bar.yaxis.set_label_position('right')
cbar.ax.tick_params(direction='in', which='both')

ax.set_xlabel('x (µm)')
ax.set_ylabel('y (µm)')
ax.set_xlim(-50, 80)
ax.set_ylim(-50, 60)

ax.hlines(y=y_values[line_index-i_range], xmin=-50, xmax=80, color='white', ls='--', lw=.8)
ax.hlines(y=y_values[line_index+i_range], xmin=-50, xmax=80, color='white', ls='--', lw=.8)

# ZOOM ON THE SELECTED LINE
fig, ax = plt.subplots( dpi=100, figsize=(5, 5) )
divider = make_axes_locatable(ax)
ax.tick_params(top=True, labeltop=True, axis='x')
ax.tick_params(bottom=False, labelbottom=False, axis='x')
ax_bar = divider.append_axes('bottom',size=.2, pad=.05)
extent_col = [xmin, xmax, y_values[line_index-i_range], y_values[line_index+i_range]]
im = ax.imshow(scan_zoom_lin, cmap='inferno', extent=extent_col, aspect='equal', norm=mpl.colors.LogNorm(vmin=1, vmax=None))

cbar = fig.colorbar(im, cax=ax_bar, label='PL intensity [kcps]', orientation='horizontal')
ax_bar.yaxis.tick_left()
ax_bar.yaxis.set_label_position('right')
cbar.ax.tick_params(direction='in', which='both')

ax.set_title("x (µm)")
ax.set_xlim(-35, 80)
ax.set_yticks([])

# fig, ax = plt.subplots( dpi=100, figsize=(6, 4) )
# ax.tick_params(direction='in', which='both')
# ax.plot(x_values, scan_lin_avg)
# ax.set_xlim(-35, 80)
pass
```

```python
# FIT LINE LINECUT WITH GAUSSIAN PEAKS

x_min = -35
x_max = 80
x_values_select, scan_lin_avg_select = dat._select_data(x_values, scan_lin_avg, x_min, x_max)
x_values_fit = np.linspace(x_values_select[0], x_values_select[-1], 1000)

df=pd.DataFrame({'x_values_select':x_values_select})
df['peaks'] = scan_lin_avg_select

N = 11
model = ConstantModel(prefix='bg_') + GaussianModel(prefix='p1_')
for i in range(2, N+1):
    model += GaussianModel(prefix='p'+str(i)+'_')

params = model.make_params(c=15,
                           p1_center=-29.7, p1_amplitude=15, p1_sigma=1,
                           p2_center=-19, p2_amplitude=300, p2_sigma=1,
                           p3_center=-9, p3_amplitude=100, p3_sigma=1,
                           p4_center=1, p4_amplitude=500, p4_sigma=1,
                           p5_center=11, p5_amplitude=800, p5_sigma=1,
                           p6_center=21, p6_amplitude=1000, p6_sigma=1,
                           p7_center=31, p7_amplitude=1300, p7_sigma=1,
                           p8_center=41, p8_amplitude=2200, p8_sigma=1,
                           p9_center=51, p9_amplitude=2800, p9_sigma=1,
                           p10_center=61, p10_amplitude=4900, p10_sigma=1,
                           p11_center=71, p11_amplitude=14400, p11_sigma=1)

params['bg_c'].min = 10
params['bg_c'].max = 20

for i in range(N):
    params['p'+str(i+1)+'_center'].min = -35
    params['p'+str(i+1)+'_center'].max = 80
    params['p'+str(i+1)+'_amplitude'].min = 15
    params['p'+str(i+1)+'_sigma'].min = .5
    params['p'+str(i+1)+'_sigma'].max = 2

# run the fit
result = model.fit(df['peaks'], params, x=df.x_values_select)

# print out the fit results
# print(result.fit_report())
```

```python
# PLOT FIT
fig, ax = plt.subplots( dpi=100, figsize=(6, 4) )
ax.tick_params(direction='in', which='both')
ax.plot(df.x_values_select, df['peaks'], marker='+', markersize=4, ls='-', color='k', label='data')
ax.plot(df.x_values_select, result.best_fit, color='r', label='fit')
plt.legend()
ax.set_yscale('log')
```

```python
#%%EXTRACT FITTED PARAMETERS

offset = result.params['bg_c'].value
offset_err = result.params['bg_c'].stderr

amplitude_fit = []
amplitude_fit_err = []
center_fit = []
center_fit_err = []
sigma_fit = []
sigma_fit_err = []
for i in range(N):
    amplitude_fit.append( result.params['p'+str(i+1)+'_amplitude'].value )
    amplitude_fit_err.append( result.params['p'+str(i+1)+'_amplitude'].stderr )
    center_fit.append( result.params['p'+str(i+1)+'_center'].value )
    center_fit_err.append( result.params['p'+str(i+1)+'_center'].stderr )
    sigma_fit.append( result.params['p'+str(i+1)+'_sigma'].value )
    sigma_fit_err.append( result.params['p'+str(i+1)+'_sigma'].stderr )
    
amplitude_fit = np.array(amplitude_fit)
amplitude_fit_err = np.array(amplitude_fit_err)
center_fit = np.array(center_fit)
center_fit_err = np.array(center_fit_err)
sigma_fit = np.array(sigma_fit)
sigma_fit_err = np.array(sigma_fit_err)

#PLOT AMPLITUDE VS PEAK INDEX
index_values = np.arange(1, 12, 1)

fig, ax = plt.subplots( dpi=100, figsize=(6, 4) )
ax.tick_params(direction='in', which='both')
ax.errorbar(index_values, amplitude_fit, yerr=amplitude_fit_err, fmt='.', color='k', ecolor='k', elinewidth=.5, capthick=.5, capsize=2)
ax.set_xlabel('Peak index')
ax.set_ylabel('Amplitude (a.u.)')
pass
```

```python
# NORMALIZED DATA

# peak_top_values = []
# for i in [1, 3, 5, 7]:
#     peak_top = amplitude_fit[i] / sigma_fit[i] / np.sqrt(2*np.pi)
#     print(peak_top)
#     peak_top_values.append(peak_top)
# norm_factor = np.mean(peak_top_values)
# print(norm_factor)

data_norm = ( df['peaks'] - offset ) / 15.734189461854825
data_fit =  ( result.best_fit - offset ) / 15.734189461854825

#PLOT
fig, ax = plt.subplots( dpi=100, figsize=(6, 4) )
ax.tick_params(direction='in', which='both')
ax.plot(df.x_values_select, data_norm, marker='+', markersize=4, ls='-', color='k', label='data')
ax.plot(df.x_values_select, data_fit, color='r', label='fit')
ax.set_xlabel('x (µm)')
ax.set_ylabel('PL intensity norm.')
ax.set_yscale('log')
ax.set_ylim(5e-1, 5e2)
ax.grid(axis='y', which='major', color='r', lw=.8, ls='-')
ax.grid(axis='y', which='minor', color='r', lw=.8, ls=':')
```

# SUBDATA SQUARE

```python
# CREATE DATAFRAME

# # col_index = range(1, 12)
# # df_amplitude=pd.DataFrame(index=col_index, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
# df_amplitude['J'] = amplitude_fit
# print(df_amplitude)

# df_amplitude.to_csv('H:/amplitude_P4_1.txt', sep='\t', index=False)
```

```python
# SELECT SQUARE
column_index = 208
line_index = 81
i_range = 10
print(x_values[column_index])
print(y_values[line_index])

scan_zoom = scan[line_index-i_range:line_index+i_range,column_index-i_range:column_index+i_range]
```

```python
#PLOT SQUARE ON RAW MAP
fig, ax = plt.subplots( dpi=100, figsize=(6, 4) )
divider = make_axes_locatable(ax)
ax_bar = divider.append_axes('right',size=.2, pad=.05)

im = ax.imshow(scan, cmap='inferno', extent=[xmin, xmax, ymin, ymax], aspect='equal', norm=mpl.colors.LogNorm(vmin=1, vmax=1e3))

cbar = fig.colorbar(im, cax=ax_bar, label='PL intensity [kcps]', orientation='vertical')
ax_bar.yaxis.tick_right()
ax_bar.yaxis.set_label_position('right')
cbar.ax.tick_params(direction='in', which='both')

ax.set_xlabel('x (µm)')
ax.set_ylabel('y (µm)')
ax.set_xlim(-50, 80)
ax.set_ylim(-50, 60)

square_x_values = [x_values[column_index-i_range], x_values[column_index+i_range], x_values[column_index+i_range], x_values[column_index-i_range], x_values[column_index-i_range]]
square_y_values = [y_values[line_index-i_range], y_values[line_index-i_range], y_values[line_index+i_range], y_values[line_index+i_range], y_values[line_index-i_range]]
ax.plot(square_x_values, square_y_values, lw=.8, ls='--', color='white')
pass
```

```python
# ZOOM ON THE SELECTED SQUARE
fig, ax = plt.subplots( dpi=100, figsize=(2, 2) )
divider = make_axes_locatable(ax)
ax_bar = divider.append_axes('right',size=.2, pad=.05)
extent_square = [x_values[column_index-i_range], x_values[column_index+i_range], y_values[line_index-i_range], y_values[line_index+i_range]]
im = ax.imshow(scan_zoom, cmap='inferno', extent=extent_square, aspect='equal')

cbar = fig.colorbar(im, cax=ax_bar, label='PL intensity [kcps]', orientation='vertical')
ax_bar.yaxis.tick_right()
ax_bar.yaxis.set_label_position('right')
cbar.ax.tick_params(direction='in', which='both')

ax.set_xlabel('x (µm)')
ax.set_ylabel('y (µm)')
# ax.set_xticks([])

print(np.mean(scan_zoom))
print(np.std(scan_zoom))
pass
```

```python
def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()
```

```python
# FIT
# Create x and y indices
x_fit = np.linspace(x_values[column_index-i_range], x_values[column_index+i_range], 2*i_range)
y_fit = np.linspace(y_values[line_index-i_range], y_values[line_index+i_range], 2*i_range)
x_fit, y_fit = np.meshgrid(x_fit, y_fit)

initial_guess = [2, -29, -28, 1, 1, 0, 1]
popt, perr, r_squared = dat._least_square(twoD_Gaussian, (x_fit, y_fit), scan_zoom.ravel(), p0=initial_guess)
print('R² = ' +str(r_squared))
print('AMPLITUDE : ' +str(popt[0]) + ' +/- ' +str(perr[0]))
print('X0 : ' +str(popt[1]) + ' +/- ' +str(perr[1]))
print('Y0 : ' +str(popt[2]) + ' +/- ' +str(perr[2]))
print('SIGMA_X : ' +str(popt[3]) + ' +/- ' +str(perr[3]))
print('SIGMA_Y : ' +str(popt[4]) + ' +/- ' +str(perr[4]))
print('THETA (RAD) : ' +str(popt[5]) + ' +/- ' +str(perr[5]))
print('THETA (DEG) : ' +str(popt[5]*57.2958) + ' +/- ' +str(perr[5]*57.2958))
print('OFFSET : ' +str(popt[6]) + ' +/- ' +str(perr[6]))
data_fitted = twoD_Gaussian((x_fit, y_fit), *popt)
```

```python
fig, ax = plt.subplots( dpi=100, figsize=(5, 5) )
divider = make_axes_locatable(ax)
ax_bar = divider.append_axes('right',size=.1, pad=.1)
ax.imshow(scan_zoom.reshape(2*i_range, 2*i_range), cmap='inferno', extent=extent_square)

cbar = fig.colorbar(im, cax=ax_bar, label='PL [kcps]', orientation='vertical')
ax_bar.yaxis.tick_right()
ax_bar.yaxis.set_label_position('right')
cbar.ax.tick_params(direction='in')

data_fitted = data_fitted.reshape(2*i_range, 2*i_range)
ax.contour(x_fit, y_fit, data_fitted, 3, colors='w')
plt.show()
```

```python

```

```python
#%%PL INTENSITY (AND NUMBER OF EMITTERS) AS A FUNCTION OF DIAMETER

file = "H:/amplitude_P4_1.txt"

#get data
amplitude = np.genfromtxt(file, skip_header=1, delimiter='\t')
amplitude_avg = np.mean(amplitude, axis=1)
amplitude_std = np.std(amplitude, axis=1)

au_to_cps_factor = 3.2 / 32
intensity_avg = amplitude_avg * au_to_cps_factor
intensity_std = amplitude_std * au_to_cps_factor

r_values = np.array([150, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 2000]) /2

# #ONE AXIS
# fig, ax = plt.subplots( dpi=300, figsize=(4, 4) )
# ax.tick_params(direction='in', which='both')
# ax.errorbar(2*r_values, intensity_avg, yerr=intensity_std, fmt='.', color='k', ecolor='k', elinewidth=.5, capthick=.5, capsize=2)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel('Hole diameter (µm)')
# ax.set_ylabel('PL intensity (kcps)')

#TWO AXIS
fig, ax_left = plt.subplots( dpi=300, figsize=(4, 4) )
ax_left.tick_params(direction='in', which='both')
ax_left.errorbar(2*r_values, intensity_avg, yerr=intensity_std, fmt='.', color='k', ecolor='k', elinewidth=.5, capthick=.5, capsize=2)
#x
ax_left.set_xscale('log')
ax_left.set_xlabel('Hole diameter (nm)')
ax_left.set_xlim(1e2, None)
#left
ax_left.set_yscale('log')
ax_left.set_ylabel('PL intensity (kcps)')
ax_left.set_ylim(2e0, None)
#right
ax_right = ax_left.twinx()
ax_right.tick_params(direction='in', which='both')
ax_right.set_yscale('log')
ax_right_ymin = ax_left.get_ylim()[0] /3.2
ax_right_ymax = ax_left.get_ylim()[1] /3.2
ax_right.set_ylim(ax_right_ymin, ax_right_ymax)
ax_right.set_ylabel('Number of emitters')
```

```python
#%%EFFECTIVE SURFACE (AND NUMBER OF IONS) AS A FUNCTION OF DIAMETER

#DEFINITION OF THE LENSE FUNCTION
def _symmetric_lense(d, R):
    term1 = 2 * R**2 * np.arccos( d/(2*R) )
    term2 = d/2 * np.sqrt(4*R**2 - d**2)
    resu = term1 - term2
    return resu

#PARAMETERS
h = 480
theta_deg = 7
theta_rad = 7 * np.pi/180
print(theta_rad)

#CALCULATE EFFECTIVE SURFACE
delta_shadow = h * np.tan(theta_rad)
print(delta_shadow)
effective_surface = _symmetric_lense(delta_shadow, r_values)
real_surface = np.pi * r_values**2
dose_cm_m2 = 1e11
dose_nm_m2 = dose_cm_m2 * 1e-14
print(effective_surface[10])
print(effective_surface[10]*dose_nm_m2)

# fig, ax = plt.subplots( dpi=300, figsize=(4, 4) )
# ax.tick_params(direction='in', which='both')
# ax.plot(2*R_values, effective_surface, marker='o', markersize=4, ls='', color='mediumvioletred')
# ax.plot(2*R_values, real_surface, ls='--', color='k')
# ax.set_xlabel('Hole diameter (nm)')
# ax.set_ylabel('Effective surface (nm$^2$)')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.s

#WITH TWO AXES
fig, ax_left = plt.subplots( dpi=300, figsize=(4, 4) )
ax_left.tick_params(direction='in', which='both')
ax_left.plot(2*r_values, effective_surface, marker='o', markersize=4, ls='', color='mediumvioletred')
ax_left.plot(2*r_values, real_surface, ls='--', color='k')
ax_left.set_xlabel('Hole diameter (nm)')
ax_left.set_ylabel('Surface (nm$^2$)')
ax_left.set_xscale('log')
ax_left.set_yscale('log')
ax_left.set_ylim(1e3, 1e7)
ax_left.set_xlim(100, 3000)
# ax_left.vlines(x=150, ymin=1e1, ymax=1e7, ls=':', color='grey')
# ax_left.hlines(y=effective_surface[10], xmin=30, xmax=3000, ls=':', color='grey')

ax_right = ax_left.twinx()
ax_right.tick_params(direction='in', which='both')
ax_right.set_yscale('log')
ax_right_ymin = ax_left.get_ylim()[0] * dose_nm_m2
ax_right_ymax = ax_left.get_ylim()[1] * dose_nm_m2
ax_right.set_ylim(ax_right_ymin, ax_right_ymax)
ax_right.set_ylabel('Average number of implanted ions')
```

```python
#%%NUMBER OF EMITTERS AS A FUNCTION OF THE NUMBER OF IONS

number_ions = effective_surface * dose_nm_m2
number_emitters = intensity_avg /3.2

def _w_vs_ions(x, I_sat, P_sat):
    resu = I_sat / ( 1 + P_sat/x )
    return resu

initial_guess = [1000, 100]
popt, perr, r_squared = stb._least_square(_w_vs_ions, number_ions, number_emitters, p0=initial_guess)
print(r_squared)
print(popt)
number_ions_fit = np.linspace(number_ions[0], number_ions[-1], 1000)
number_emitters_fit = _w_vs_ions(number_ions_fit, popt[0], popt[1])
print(popt[0]/popt[1])

fig, ax = plt.subplots( dpi=300, figsize=(4, 4) )
ax.tick_params(direction='in', which='both')
ax.plot(number_ions, number_emitters, marker='o', markersize=4, ls='', color='r')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of ions')
ax.set_ylabel('Number of emitters')

ax.plot(number_ions_fit, number_emitters_fit, color='k')
```
