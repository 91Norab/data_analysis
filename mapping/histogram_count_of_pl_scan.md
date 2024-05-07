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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz
from pathlib import Path
from matplotlib.colors import LogNorm
import data_analysis_toolbox as dat
import lmfit
```

```python
import pandas as pd
pd.options.display.max_rows = 150
pd.options.display.max_columns = 150
```

```python
from ast import literal_eval
def str_to_tuple(s):
    try:
        return literal_eval(s)
    except (ValueError, SyntaxError) as e:
        return None
```

```python
folder = r"C:\Users\YB274940\qudi\Data\2024\04\2024-04-04"
data = dat.get_dataframe_from_folders(folder)
```

```python
data
```

```python
def params_for_plot(data, data_line):
    v_min, v_max = 0.3, 1000
    
    x_min = data[" x scan range"].apply(str_to_tuple)[data_line][0]
    x_max = data[" x scan range"].apply(str_to_tuple)[data_line][1]
    y_min = data[" y scan range"].apply(str_to_tuple)[data_line][0]
    y_max = data[" y scan range"].apply(str_to_tuple)[data_line][1]
    extent = (x_min, x_max, y_min, y_max)
    
    scalebar_x_max = x_max - x_max*0.2
    scalebar_x_min = scalebar_x_max - 5e-6
    scalebar_y_max = y_min - y_min*0.2
    scalebar_y_min = scalebar_y_max
    scalebar_text_x = scalebar_x_min + scalebar_x_min*0.05
    scalebar_text_y = scalebar_y_min - scalebar_y_min*0.1
    scalebar_text_fontsize = 6 if x_max == 20e-6 else 6
    
    params_dict = {
        'v_min': v_min,
        'v_max': v_max,
        'x_min': x_min,
        'y_max':y_max,
        'extent': extent,
        'scalebar_x_max': scalebar_x_max,
        'scalebar_x_min': scalebar_x_min,
        'scalebar_y_max': scalebar_y_max,
        'scalebar_y_min': scalebar_y_min,
        'scalebar_text_x': scalebar_text_x,
        'scalebar_text_y': scalebar_text_y,
        'scalebar_text_fontsize': scalebar_text_fontsize
    }
    
    return params_dict

def plot_line_from_dataframe(ax, data_line, params_dict):
    data_to_plot = data["_raw_data"][data_line][::-1] / 1e3

    im = ax.imshow(data_to_plot, cmap='inferno', extent=params_dict['extent'], 
#                    vmin=params_dict['v_min'], vmax=params_dict['v_max'], )
                   norm=LogNorm( vmin=params_dict['v_min'], vmax=params_dict['v_max']))
    ax.plot([params_dict['scalebar_x_min'], params_dict['scalebar_x_max']], 
            [params_dict['scalebar_y_min'], params_dict['scalebar_y_max']], color='white', linewidth=2)
    ax.text(params_dict['scalebar_text_x'], params_dict['scalebar_text_y'], 
            '5 µm', color='white', fontsize=params_dict['scalebar_text_fontsize'])

    ax.set_xticks([])
    ax.set_yticks([])

    return im
```

```python
data_line = 3
text_to_plot = '1e10 Si/cm² & 250 °C'
params_dict = params_for_plot(data, data_line)

fig, axs = plt.subplots(figsize=(4,4))
im = plot_line_from_dataframe(axs, data_line, params_dict)
axs.text(params_dict['x_min']+25e-6, params_dict['y_max']-2e-6, text_to_plot, 
         color='black', bbox=dict(facecolor='white', alpha=0.6), fontsize=6)

cbar = fig.colorbar(im, ax=axs, fraction=0.045, shrink= 1.0, aspect=20, pad=0.05)
cbar.set_label('PL intensity [kc/s]')

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.tight_layout(pad=2.0)

# plt.savefig('20240403_T2A.png')
plt.show()
fig.dpi=200
pass
```

```python
data_to_plot = data["_raw_data"][data_line][::-1]
scan = data_to_plot

print(np.mean(scan), np.std(scan), np.std(scan)/np.mean(scan))
```

```python
# t = row['Time (s)']
y = scan.flatten()
```

```python
y
```

```python
color = 'dodgerblue'
rebin = 1
```

```python
np.std(y)
```

```python
np.sqrt(np.mean(y))
```

```python
def gaussian_function(params, x, data):
    model = params['offset']
    model += params['amplitude'] *np.exp(-1/2*((x-params['x0'])/params['sigma'])**2)
    return model - data

params_gaus = lmfit.Parameters()
params_gaus.add('offset', value=0)
params_gaus.add('amplitude', value=100)
params_gaus.add('x0', value=30000)
params_gaus.add('sigma', value=20000)
```

```python
params_poisson = lmfit.Parameters()
params_poisson.add('offset', value=0, vary=False)
params_poisson.add('amplitude', value=100, vary=False)
params_poisson.add('x0', value=np.mean(y), vary=False)
params_poisson.add('sigma', value=np.sqrt(np.mean(y)), vary=False)
```

```python
def plot_hist_ph_count(ax):
    ax.set_xlabel("Photons per bin [#]", labelpad=0)
    ax.set_ylabel("Events [k#]", labelpad=0)
    
    h = ax.hist(y, bins = 70, range=[0,50000], edgecolor = 'white', linewidth = 0.3
                , color = 'dodgerblue', histtype='bar',  alpha = 0.6)

    x_fit = np.linspace(0, 50000, 1000)
    result = lmfit.minimize(gaussian_function, params_gaus, args=(h[1][:-1], h[0]))
    bin_shift = h[1][1]-h[1][0]
    y_fit = gaussian_function(result.params, x_fit, 0)
    ax.plot(x_fit+bin_shift/2, y_fit, color='blue', ls='--', alpha=1)
    
    result_poisson = lmfit.minimize(gaussian_function, params_poisson, args=(h[1][:-1], h[0]))
    y_poisson = gaussian_function(result_poisson.params, x_fit, 0)
    ax.plot(x_fit, y_poisson, color='red', ls='-', alpha=1)
    
#     ax.set_xlim(0,None)
#     ax.set_ylim(0,300)
#     ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([0,20000]))
#     ax.set_yticklabels([0,20])
#     ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([5000,10000,15000]))
#     ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([20,60,100]))
#     ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator([30,40,50,70,80,90]))
    

fig, ax = plt.subplots(figsize=(4,3))
plot_hist_ph_count(ax)
fig.dpi=200
```

```python

```
