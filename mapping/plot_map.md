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
import matplotlib.pyplot as plt
import data_analysis_toolbox as dat
from pathlib import Path
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
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

```python
import pandas as pd
pd.options.display.max_rows = 150
pd.options.display.max_columns = 150
```

---
# DATA

```python
folder = r"C:\Users\YB274940\qudi\Data\2024\03\2024-03-28"
data = dat.get_dataframe_from_folders(folder)
```

```python
data
```

---
# PLOT

```python
def params_for_plot(data, data_line):
    v_min, v_max = 0.1, 200
    
    x_scan_range = data[" x scan range"].apply(str_to_tuple)
    x_min = x_scan_range[data_line][0]
    x_max = x_scan_range[data_line][1]
    
    y_scan_range = data[" y scan range"].apply(str_to_tuple)
    y_min = y_scan_range[data_line][0]
    y_max = y_scan_range[data_line][1]
    
    extent = (x_min, x_max, y_min, y_max)
    
    scalebar_x_max = x_max - 185e-6
    scalebar_x_min = scalebar_x_max - 10e-6
    scalebar_y_max = y_min - y_min * 0.15
    scalebar_y_min = scalebar_y_max
    scalebar_text_x = scalebar_x_min
    scalebar_text_y = scalebar_y_min - scalebar_y_min * 0.05
    scalebar_text_fontsize = 12
    
    params_dict = {
        'v_min': v_min,
        'v_max': v_max,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
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
```

```python
def plot_line_from_dataframe(ax, data_line, params_dict):
    data_to_plot = data["_raw_data"][data_line][::-1] / 1e3
    
    # Add a small offset to avoid zeros in the data
    offset = 1e-10
    data_to_plot = np.where(data_to_plot == 0, offset, data_to_plot)
    
    # Plot 2D scan
    im = ax.imshow(data_to_plot, cmap='inferno', extent=params_dict['extent'], 
#                    vmin=params_dict['v_min'], vmax=params_dict['v_max'], )
                   norm=LogNorm( vmin=params_dict['v_min'], vmax=params_dict['v_max']))
    
    # Plot scale bar
    ax.plot([params_dict['scalebar_x_min'], params_dict['scalebar_x_max']], 
            [params_dict['scalebar_y_min'], params_dict['scalebar_y_max']], 
            color='white', linewidth=2)
    
    # Add scale bar text
    ax.text(params_dict['scalebar_text_x'], params_dict['scalebar_text_y'], 
            '10 µm', color='white', fontsize=params_dict['scalebar_text_fontsize'])
    
    
# =================================================
#     data_bg = data_to_plot[30:360,0:50]
#     print(data_bg.mean())
    
#     # Create a Rectangle patch
#     rect = Rectangle((params_dict['x_min'], params_dict['y_min']+30e-6*0.333), 50e-6*0.333, 330e-6*0.333, 
#                      linewidth=1, edgecolor='r', facecolor='none')
#     # Add the patch to the Axes
#     ax.add_patch(rect)
# =================================================
    
    
    ax.set_xticks([])
    ax.set_yticks([])

    return im
```

```python
data_line = 0
params_dict = params_for_plot(data, data_line)

fig, axs = plt.subplots(figsize=(8,6))
im = plot_line_from_dataframe(axs, data_line, params_dict)

cbar = fig.colorbar(im, orientation='horizontal', ax=axs, fraction=0.045, shrink=1.0, aspect=20, pad=0.05)
cbar.set_label('PL intensity [kc/s]')

fig.dpi=200
pass
```

```python
# for i in range(10):
#     fig, axs = plt.subplots(figsize=(4,4))
#     im = plot_line_from_dataframe(axs, i, params_dict)

#     cbar = fig.colorbar(im, ax=axs, fraction=0.045, shrink= 1.0, aspect=20, pad=0.05)
#     cbar.set_label('PL intensity [kc/s]')
#     plt.show()
#     fig.dpi=200
    
# pass
```

```python
# fig, axs = plt.subplots(2, 2)

# ims = {}
# for i, (row, col, data_line, text_to_plot) in enumerate([(0, 0, 0, '1e11 Si/cm²')
#                                                          , (0, 1, 1, '1e10 Si/cm²')
#                                                          , (1, 0, 3, '1e09 Si/cm²')
#                                                          , (1, 1, 5, '1e08 Si/cm²')]):
#     params_dict = params_for_plot(data, data_line)
#     im = plot_line_from_dataframe(axs[row, col], data_line, params_dict)
#     axs[row, col].text(params_dict['x_min']+1e-6, params_dict['y_max']-2e-6, text_to_plot, color='black', bbox=dict(facecolor='white', alpha=0.6), fontsize=6)
#     ims[(row, col)] = im

# # Create a common colorbar based on the last plot's colormap
# cbar = fig.colorbar(ims[(1, 1)], ax=axs, fraction=0.045, shrink=1.0, aspect=20, pad=0.05, )
# cbar.set_label('PL intensity [kc/s]')

# fig.suptitle('T = 300 °C', x=0.52, y=1.0, fontsize=8)
# fig.dpi = 200
# pass
```

```python
# fig, axs = plt.subplots(1, 4, figsize=(8,4))

# ims = {}
# for i, (row, col, data_line, text_to_plot) in enumerate([(0, 0, 1, '1e11 Si/cm²')
#                                                          , (0, 1, 3, '1e10 Si/cm²')
#                                                          , (0, 2, 4, '1e09 Si/cm²')
#                                                          , (0, 3, 5, '1e08 Si/cm²')]):
#     params_dict = params_for_plot(data, data_line)
#     im = plot_line_from_dataframe(axs[col], data_line, params_dict)
#     axs[col].text(params_dict['x_min']+1e-6, params_dict['y_max']-2e-6, text_to_plot, color='black', bbox=dict(facecolor='white', alpha=0.6), fontsize=6)
#     ims[col] = im

# # Create a common colorbar based on the last plot's colormap
# cbar = fig.colorbar(ims[3], ax=axs, fraction=0.045, shrink=1.0, aspect=20, pad=0.05, )
# cbar.set_label('PL intensity [kc/s]')

# fig.suptitle('T = 300 °C', x=0.52, y=1.0, fontsize=8)
# fig.dpi = 200
# pass
```

```python
# fig, axs = plt.subplots(1, 4, figsize=(8, 4))

# ims = {}
# for i, (row, col, data_line, text_to_plot) in enumerate([(0, 0, 0, 'T$_{anneal}$ = 200 °C'),
#                                                          (0, 1, 1, 'T$_{anneal}$ = 250 °C'),
#                                                          (0, 2, 2, 'T$_{anneal}$ = 300 °C'),
#                                                          (0, 3, 4, 'T$_{anneal}$ = 350 °C')]):
#     params_dict = params_for_plot(data, data_line)
#     im = plot_line_from_dataframe(axs[col], data_line, params_dict)
#     axs[col].text(params_dict['x_min'] + 0.6*params_dict['x_max'], params_dict['y_max'] - 0.11*params_dict['y_max'], 
#                   text_to_plot, color='black', bbox=dict(facecolor='white', alpha=0.6), fontsize=6)
#     ims[col] = im

# # Create a common colorbar based on the last plot's colormap
# cbar = fig.colorbar(ims[3], orientation='horizontal', ax=axs, fraction=0.045, shrink=1.0, aspect=20, pad=0.05)
# cbar.set_label('PL intensity [kc/s]')

# fig.suptitle('Fluence: 1e11 Si/cm² at Room-T', x=0.5, y=0.62, fontsize=8)
# fig.dpi = 200
# plt.show()
```
