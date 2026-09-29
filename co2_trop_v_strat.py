import argparse
import xarray as xr
import numpy as np
from helpers import log_args, convert_ppm, file_flat, file_zec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors


def plot_co2_monthly_trend(data, years, zec=False):
    """
    Plots the continuous monthly global mean CO2 concentrations over time,
    comparing the Troposphere and Stratosphere without integer-year stepping.
    """
    # 1. Collapse spatial dimensions
    global_profile = data["CO2"].mean(dim=['lat', 'lon'])
    
    # 2. Separate vertical pressure levels (using 'plev' in Pascals)
    p_levels = global_profile['plev'].values
    trop_mask = (p_levels <= 100000) & (p_levels >= 20000)
    strat_mask = (p_levels < 20000) & (p_levels >= 1000)
    
    # 3. Calculate layer averages and convert to ppm values
    troposphere_mean_raw = global_profile.isel(plev=trop_mask).mean(dim='plev')
    stratosphere_mean_raw = global_profile.isel(plev=strat_mask).mean(dim='plev')
    
    troposphere_mean = convert_ppm(troposphere_mean_raw, key=None)
    stratosphere_mean = convert_ppm(stratosphere_mean_raw, key=None)

    # 4. Extract the raw datetime timeline array
    # This keeps every month distinct instead of collapsing them into integer years
    time_axis = troposphere_mean.time.values

    # 5. Plotting Time Trends
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Passing the raw datetime array fixes the staircase pattern natively
    ax.plot(time_axis, troposphere_mean.values, color='#1f77b4', linewidth=2, label='Troposphere Average (Surface - 200 hPa)')
    ax.plot(time_axis, stratosphere_mean.values, color='#ff7f0e', linewidth=2, linestyle='--', label='Stratosphere Average (200 - 10 hPa)')
    
    # 6. Advanced X-Axis Date Formatting
    # This automatically adds clean year tick labels to the horizontal axis
    ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Puts a major tick mark every 5 years
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Formats the label as just the Year
    
    # Formatting
    ax.set_title("Global CO2 Monthly Accumulation Trends: Troposphere vs. Stratosphere", fontsize=14, fontweight='bold', pad=15)
    if zec:
        ax.set_title("Global CO2 Monthly Accumulation Trends (ZEC): Troposphere vs. Stratosphere", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Atmospheric CO2 Concentration (ppm)", fontsize=12)
    ax.set_xlabel("Model Year", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc="upper left", fontsize=11)
    
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)
    plt.show()


def plot_co2_zonal_mean_trend(data, years, zec=False):
    """
    Plots the continuous monthly mean CO2 concentrations over time,
    comparing different latitude zones (zonal means).
    """
    # 1. Collapse only longitude and pressure levels to focus on latitude over time
    # (Assuming you want a full-column or deep-layer average for each latitude band)
    zonal_profile = data["CO2"].mean(dim=['lon', 'plev'])
    
    # 2. Define your latitude zones using masks
    # Adjust the boundaries (e.g., -30 to 30 for Tropics) based on your needs
    lats = zonal_profile['lat'].values
    nh_mask = (lats >= 30)
    trop_mask = (lats > -30) & (lats < 30)
    sh_mask = (lats <= -30)
    
    # 3. Calculate spatial zone averages and convert to ppm values
    nh_mean_raw = zonal_profile.isel(lat=nh_mask).mean(dim='lat')
    trop_mean_raw = zonal_profile.isel(lat=trop_mask).mean(dim='lat')
    sh_mean_raw = zonal_profile.isel(lat=sh_mask).mean(dim='lat')
    
    nh_mean = convert_ppm(nh_mean_raw, key=None)
    trop_mean = convert_ppm(trop_mean_raw, key=None)
    sh_mean = convert_ppm(sh_mean_raw, key=None)

    # 4. Extract the raw datetime timeline array
    time_axis = nh_mean.time.values

    # 5. Plotting Zonal Time Trends
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time_axis, nh_mean.values, linewidth=2, label='Northern Hemisphere (30°N - 90°N)')
    ax.plot(time_axis, trop_mean.values, linewidth=2, label='Tropics (30°S - 30°N)')
    ax.plot(time_axis, sh_mean.values, linewidth=2, label='Southern Hemisphere (90°S - 30°S)')
    
    # 6. Advanced X-Axis Date Formatting
    ax.xaxis.set_major_locator(mdates.YearLocator(5))  
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 
    
    # Formatting
    ax.set_title("Global CO2 Monthly Accumulation Trends by Latitude Zone", fontsize=14, fontweight='bold', pad=15)
    if zec:
        ax.set_title("Global CO2 Monthly Accumulation Trends (ZEC): by Latitude Zone", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Atmospheric CO2 Concentration (ppm)", fontsize=12)
    ax.set_xlabel("Model Year", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc="upper left", fontsize=11)
    
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)
    plt.show()


def plot_co2_zonal_and_vertical_trend(data, years, zec=False):
    """
    Plots continuous monthly mean CO2 concentrations over time,
    splitting by BOTH vertical layers (Troposphere vs. Stratosphere)
    and latitude zones (NH, Tropics, SH) with maximally distinct colors.
    """
    # 1. Collapse only longitude to preserve both pressure (plev) and latitude (lat)
    spatial_profile = data["CO2"].mean(dim=['lon'])
    
    # 2. Extract dimension arrays
    p_levels = spatial_profile['plev'].values
    lats = spatial_profile['lat'].values
    
    # 3. Define masks for vertical layers and horizontal zones
    trop_mask = (p_levels <= 100000) & (p_levels >= 20000)
    strat_mask = (p_levels < 20000) & (p_levels >= 1000)
    
    nh_mask = (lats >= 30)
    trop_lat_mask = (lats > -30) & (lats < 30)
    sh_mask = (lats <= -30)
    
    # 4. Extract data slices and calculate averages
    # --- TROPOSPHERE ZONAL MEANS ---
    trop_layer = spatial_profile.isel(plev=trop_mask).mean(dim='plev')
    trop_nh = convert_ppm(trop_layer.isel(lat=nh_mask).mean(dim='lat'), key=None)
    trop_trop = convert_ppm(trop_layer.isel(lat=trop_lat_mask).mean(dim='lat'), key=None)
    trop_sh = convert_ppm(trop_layer.isel(lat=sh_mask).mean(dim='lat'), key=None)
    
    # --- STRATOSPHERE ZONAL MEANS ---
    strat_layer = spatial_profile.isel(plev=strat_mask).mean(dim='plev')
    strat_nh = convert_ppm(strat_layer.isel(lat=nh_mask).mean(dim='lat'), key=None)
    strat_trop = convert_ppm(strat_layer.isel(lat=trop_lat_mask).mean(dim='lat'), key=None)
    strat_sh = convert_ppm(strat_layer.isel(lat=sh_mask).mean(dim='lat'), key=None)

    # 5. Extract time axis
    time_axis = trop_nh.time.values

    # 6. Plotting - Using 2 subplots for clarity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Maximally distinct custom color palette (No Blue/Green overlapping colors)
    colors = {
        'NH': '#D32F2F',       # Crimson Red
        'Tropics': '#212121',  # Dark Charcoal / Off-Black
        'SH': '#FF9800'        # Vivid Amber / Orange-Gold
    }
    
    # --- Panel 1: Troposphere ---
    ax1.plot(time_axis, trop_nh.values, color=colors['NH'], linewidth=2, label='Northern Hemisphere (30°N - 90°N)')
    ax1.plot(time_axis, trop_trop.values, color=colors['Tropics'], linewidth=2, label='Tropics (30°S - 30°N)')
    ax1.plot(time_axis, trop_sh.values, color=colors['SH'], linewidth=2, label='Southern Hemisphere (90°S - 30°S)')
    ax1.set_title("Troposphere Average (Surface - 200 hPa)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("CO2 (ppm)", fontsize=11)
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.legend(loc="upper left", fontsize=10, frameon=True, edgecolor='#CCCCCC')
    
    # --- Panel 2: Stratosphere ---
    ax2.plot(time_axis, strat_nh.values, color=colors['NH'], linewidth=2, label='Northern Hemisphere (30°N - 90°N)')
    ax2.plot(time_axis, strat_trop.values, color=colors['Tropics'], linewidth=2, label='Tropics (30°S - 30°N)')
    ax2.plot(time_axis, strat_sh.values, color=colors['SH'], linewidth=2, label='Southern Hemisphere (90°S - 30°S)')
    ax2.set_title("Stratosphere Average (200 - 10 hPa)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("CO2 (ppm)", fontsize=11)
    ax2.set_xlabel("Model Year", fontsize=11)
    ax2.grid(True, alpha=0.25, linestyle='--')
    ax2.legend(loc="upper left", fontsize=10, frameon=True, edgecolor='#CCCCCC')
    
    # 7. Formatting X-Axis
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))  
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 
    
    fig.suptitle("Global CO2 Monthly Trends: Zonal Breakdown by Atmospheric Layer", fontsize=14, fontweight='bold', y=0.96)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.88, hspace=0.25)
    plt.show()


def plot_co2_zonal_grid_trend(data, years, layer='troposphere', zec=False):
    """
    Plots monthly mean CO2 concentrations over time for EVERY latitude grid point.
    Splits the results into side-by-side graphs with perfectly synchronized colorbars.
    """
    # 1. Collapse only longitude to preserve full latitude and pressure structures
    spatial_profile = data["CO2"].mean(dim=['lon'])
    p_levels = spatial_profile['plev'].values
    
    # 2. Dynamically apply the vertical layer selection mask
    layer_str = layer.lower()
    if layer_str == 'troposphere':
        v_mask = (p_levels <= 100000) & (p_levels >= 20000)
        title_layer = "Troposphere (Surface - 200 hPa)"
    elif layer_str == 'stratosphere':
        v_mask = (p_levels < 20000) & (p_levels >= 1000)
        title_layer = "Stratosphere (200 - 10 hPa)"
    else:
        raise ValueError("Invalid layer input. Choose either 'troposphere' or 'stratosphere'.")
        
    # 3. Compute zonal mean for each individual latitude coordinate
    zonal_grid_raw = spatial_profile.isel(plev=v_mask).mean(dim='plev')
    zonal_grid = convert_ppm(zonal_grid_raw, key=None)
    
    # 4. Extract coordinate structures
    time_axis = zonal_grid.time.values
    lats = zonal_grid['lat'].values

    # 5. Initialize Side-by-Side subplots
    fig, (ax_nh, ax_sh) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    # Truncate colormaps to start at 0.3 (light pink/light purple) instead of 0.0 (pure white)
    # This keeps lines highly visible AND perfectly synchronizes them with the colorbars
    cmap_nh = mcolors.LinearSegmentedColormap.from_list(
        'trunc_reds', plt.cm.Reds(np.linspace(0.3, 1.0, 256))
    )
    cmap_sh = mcolors.LinearSegmentedColormap.from_list(
        'trunc_purples', plt.cm.Purples(np.linspace(1.0, 0.3, 256)) # Inverted so poles are dark
    )
    
    norm_nh = mcolors.Normalize(vmin=0, vmax=90)
    norm_sh = mcolors.Normalize(vmin=-90, vmax=0)

    # 6. Loop over every grid point and plot
    nh_count = np.sum(lats >= 0)
    sh_count = np.sum(lats < 0)
    
    for i, lat_val in enumerate(lats):
        line_data = zonal_grid.isel(lat=i).values
        
        # Northern Hemisphere
        if lat_val >= 0:
            color_val = cmap_nh(norm_nh(lat_val))
            ax_nh.plot(time_axis, line_data, color=color_val, linewidth=1.2)
            
        # Southern Hemisphere
        else:
            color_val = cmap_sh(norm_sh(lat_val))
            ax_sh.plot(time_axis, line_data, color=color_val, linewidth=1.2)

    # 7. Axes Formatting
    for ax in [ax_nh, ax_sh]:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))  
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xlabel("Model Year", fontsize=11)
        
    ax_nh.set_ylabel("Atmospheric CO2 Concentration (ppm)", fontsize=12)
    ax_nh.set_title(f"Northern Hemisphere Grid Points\n(0° to 90°N — {nh_count} latitudinal rows)", fontsize=12, fontweight='bold')
    ax_sh.set_title(f"Southern Hemisphere Grid Points\n(90°S to 0°S — {sh_count} latitudinal rows)", fontsize=12, fontweight='bold')
    
    # 8. Add Colorbars (Now perfectly matching the plotted lines)
    sm_nh = plt.cm.ScalarMappable(cmap=cmap_nh, norm=norm_nh)
    cbar_nh = fig.colorbar(sm_nh, ax=ax_nh, orientation='vertical', pad=0.03, shrink=0.85)
    cbar_nh.set_label('Latitude (°N)', fontsize=11)
    
    sm_sh = plt.cm.ScalarMappable(cmap=cmap_sh, norm=norm_sh)
    cbar_sh = fig.colorbar(sm_sh, ax=ax_sh, orientation='vertical', pad=0.03, shrink=0.85)
    # Reverse the ticks on the SH colorbar to visually go from 0 down to -90 logically
    cbar_sh.ax.invert_yaxis()
    cbar_sh.set_label('Latitude (°S / Negative)', fontsize=11)

    fig.suptitle(f"Zonal Grid Point Evolution Trends: {title_layer}", fontsize=15, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.12, top=0.86, wspace=0.22)
    fig.tight_layout()
    plt.show()


def plot_co2_vertical_grid_trend(data, years, zec=False):
    """
    Plots monthly mean CO2 concentrations over time for EVERY individual pressure level.
    Splits the results into side-by-side graphs with perfectly synchronized log-pressure colorbars.
    """
    # 1. Collapse only longitude to preserve full latitude and pressure structures
    spatial_profile = data["CO2"].mean(dim=['lon'])
    
    # 2. Separate into Northern and Southern Hemisphere spatial arrays first
    lats = spatial_profile['lat'].values
    nh_profile = spatial_profile.isel(lat=(lats >= 0)).mean(dim='lat')
    sh_profile = spatial_profile.isel(lat=(lats < 0)).mean(dim='lat')
    
    # 3. Convert raw values to ppm
    nh_grid = convert_ppm(nh_profile, key=None)
    sh_grid = convert_ppm(sh_profile, key=None)
    
    # 4. Extract coordinate structures (converting pressure from Pa to hPa)
    time_axis = nh_grid.time.values
    p_levels_hpa = nh_grid['plev'].values / 100.0  # Convert Pascals to Hectopascals

    # 5. Initialize Side-by-Side subplots
    fig, (ax_nh, ax_sh) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    # Fix: Reverse the color mapping array itself (1.0 to 0.3) 
    # This maps low values (0.1 hPa, top of atmosphere) to DARK colors
    # and high values (1000 hPa, surface) to LIGHT colors automatically
    cmap_nh = mcolors.LinearSegmentedColormap.from_list(
        'trunc_reds_vert', plt.cm.Reds(np.linspace(1.0, 0.3, 256))
    )
    cmap_sh = mcolors.LinearSegmentedColormap.from_list(
        'trunc_purples_vert', plt.cm.Purples(np.linspace(1.0, 0.3, 256))
    )
    
    # Logarithmic normalization matching exponential atmospheric pressure drop
    norm_press = mcolors.LogNorm(vmin=p_levels_hpa.min(), vmax=p_levels_hpa.max())

    # 6. Loop over every vertical pressure level grid point and plot
    plev_count = len(p_levels_hpa)
    
    for i, p_val in enumerate(p_levels_hpa):
        # The colorbar and lines now use the identical matching normalization engine
        color_val_nh = cmap_nh(norm_press(p_val))
        color_val_sh = cmap_sh(norm_press(p_val))
        
        # Plot Northern Hemisphere profile line
        line_nh = nh_grid.isel(plev=i).values
        ax_nh.plot(time_axis, line_nh, color=color_val_nh, linewidth=1.2)
        
        # Plot Southern Hemisphere profile line
        line_sh = sh_grid.isel(plev=i).values
        ax_sh.plot(time_axis, line_sh, color=color_val_sh, linewidth=1.2)

    # 7. Axes Formatting
    for ax in [ax_nh, ax_sh]:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))  
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xlabel("Model Year", fontsize=11)
        
    ax_nh.set_ylabel("Atmospheric CO2 Concentration (ppm)", fontsize=12)
    ax_nh.set_title(f"Northern Hemisphere Profiles\n(0° to 90°N — {plev_count} vertical levels)", fontsize=12, fontweight='bold')
    ax_sh.set_title(f"Southern Hemisphere Profiles\n(90°S to 0°S — {plev_count} vertical levels)", fontsize=12, fontweight='bold')
    
    # 8. Add Colorbars (Now perfectly matching the plotted lines)
    sm_nh = plt.cm.ScalarMappable(cmap=cmap_nh, norm=norm_press)
    cbar_nh = fig.colorbar(sm_nh, ax=ax_nh, orientation='vertical', pad=0.03, shrink=0.85)
    cbar_nh.ax.invert_yaxis()  # Puts 1000 hPa (Surface) at the bottom
    cbar_nh.set_label('Pressure Level (hPa)', fontsize=11)
    
    sm_sh = plt.cm.ScalarMappable(cmap=cmap_sh, norm=norm_press)
    cbar_sh = fig.colorbar(sm_sh, ax=ax_sh, orientation='vertical', pad=0.03, shrink=0.85)
    cbar_sh.ax.invert_yaxis()  # Puts 1000 hPa (Surface) at the bottom
    cbar_sh.set_label('Pressure Level (hPa)', fontsize=11)

    fig.suptitle("Vertical Grid Point Evolution Trends: Full-Column Profiles", fontsize=15, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.12, top=0.86, wspace=0.22)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_args(args)

    path = file_flat
    if (args.zec):
        path = file_zec
    print(path)
    years = np.arange(1900, 1925, 1, dtype=int)
    # print(years)
    # months = [9] # np.arange(1, 13, dtype=int)
    # print(months)

    data = xr.open_dataset(path)
    data = data.sel(time=slice(f"{years[0]}-01-01", f"{years[-1]}-12-31"))

    # plot_co2_monthly_trend(data, years, zec=args.zec)
    # plot_co2_zonal_mean_trend(data, years, zec=args.zec)
    # plot_co2_zonal_and_vertical_trend(data, years, zec=args.zec)
    plot_co2_zonal_grid_trend(data, years, layer='troposphere', zec=args.zec)
    plot_co2_zonal_grid_trend(data, years, layer='stratosphere', zec=args.zec)
    plot_co2_vertical_grid_trend(data, years, zec=args.zec)
    