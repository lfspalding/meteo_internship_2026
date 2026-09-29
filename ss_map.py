import argparse
import xarray as xr
import numpy as np
from helpers import log_args, convert_ppm, map_setup, remove_seasonal_cycle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
from scipy import signal


class CoordinateBox:
    def __init__(self, name, lat0, lat1, lon0, lon1):
        self.name = name
        self.lat0 = lat0
        self.lat1 = lat1
        self.lon0 = lon0
        self.lon1 = lon1

    def __repr__(self):
        return f"CoordinateBox(name={self.name}, lat0={self.lat0}, lat1={self.lat1}, lon0={self.lon0}, lon1={self.lon1})"


def plot_co2_anomalies_seasonal_maps(data, year_to_plot=1900):
    """
    Plots side-by-side geographic maps showing the winter source intensity 
    and summer sink intensity footprints for a given year.
    """
    # 1. Calculate the raw Global Mean over time
    global_mean = data["CO2"].mean(dim=['lat', 'lon'])
    
    # 2. STRIP THE BACKGROUND MIXING across the entire 2D grid
    regional_anomaly_raw = data["CO2"] - global_mean
    
    # Convert the anomaly array to ppm unit values
    regional_anomaly = convert_ppm(regional_anomaly_raw, key=None)
    
    # 3. Filter for the specific year
    yearly_data = regional_anomaly.sel(time=str(year_to_plot))
    
    winter_months = yearly_data['time.month'].isin([12, 1, 2])
    winter_source = yearly_data.where(winter_months, drop=True).mean(dim='time')
    summer_months = yearly_data['time.month'].isin([6, 7, 8])
    summer_sink = yearly_data.where(summer_months, drop=True).mean(dim='time')

    # 5. Set up the figure with 1 row and 2 columns using a Map Projection
    fig, axes = plt.subplots(
        1, 2, 
        figsize=(18, 6.5), 
        subplot_kw={'projection': ccrs.Robinson()}
    )
    
    # Define common plotting configurations
    map_features = [
        (cfeature.COASTLINE, 1.0, 'black'),
        (cfeature.BORDERS, 0.5, 'gray')
    ]
    
    ax_win = axes[0]
    for feature, lw, color in map_features:
        ax_win.add_feature(feature, linewidth=lw, edgecolor=color)
    ax_win.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    
    im_win = winter_source.plot(
        ax=ax_win, 
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r', 
        center=0, 
        # robust=True,
        vmin=-1, vmax=1,  # Set fixed color limits for better comparison
        cbar_kwargs={'label': 'Anomaly (ppm)', 'orientation': 'horizontal', 'pad': 0.05, 'shrink': 0.8}
    )
    ax_win.set_title(f"DJF Magnitude ({year_to_plot})", fontsize=12, fontweight='bold')
    gl_win = ax_win.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl_win.top_labels, gl_win.right_labels = False, False

    ax_sum = axes[1]
    for feature, lw, color in map_features:
        ax_sum.add_feature(feature, linewidth=lw, edgecolor=color)
    ax_sum.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    
    im_sum = summer_sink.plot(
        ax=ax_sum, 
        transform=ccrs.PlateCarree(),
        cmap='PiYG_r', 
        center=0, 
        # robust=True,
        vmin=-1, vmax=1,  # Set fixed color limits for better comparison
        cbar_kwargs={'label': 'Anomaly (ppm)', 'orientation': 'horizontal', 'pad': 0.05, 'shrink': 0.8}
    )
    ax_sum.set_title(f"JJA Magnitude ({year_to_plot})", fontsize=12, fontweight='bold')
    gl_sum = ax_sum.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl_sum.top_labels, gl_sum.left_labels = False, False  # Hide left labels on the second plot to avoid overlap

    # Global adjustments
    plt.suptitle(f"Carbon Footprint Anomalies (Relative to Yearly Global Mean)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_co2_anomalies_vs_year(data, year_to_plot=1900):
    """
    Plots side-by-side geographic maps showing the winter source intensity 
    and summer sink intensity footprints for a given year.
    """
    # 1. Filter for the specific year FIRST
    yearly_data_raw = data["CO2"].sel(time=str(year_to_plot))

    # 2. Calculate a single STATIC scalar baseline for that entire year
    # This collapses lat, lon, AND time into one constant number (e.g., 350.0 ppm)
    annual_global_baseline = yearly_data_raw.mean(dim=['lat', 'lon', 'time'])
    
    # 3. STRIP THE BACKGROUND MIXING using the flat benchmark
    # This preserves the seasonal drop in summer without subtracting it away
    regional_anomaly_raw = yearly_data_raw - annual_global_baseline
    
    # Convert the anomaly array to ppm unit values
    yearly_data = convert_ppm(regional_anomaly_raw, key=None)
    
    winter_months = yearly_data['time.month'].isin([12, 1, 2])
    winter_source = yearly_data.where(winter_months, drop=True).mean(dim='time')
    summer_months = yearly_data['time.month'].isin([6, 7, 8])
    summer_sink = yearly_data.where(summer_months, drop=True).mean(dim='time')

    # 5. Set up the figure with 1 row and 2 columns using a Map Projection
    fig, axes = plt.subplots(
        1, 2, 
        figsize=(18, 6.5), 
        subplot_kw={'projection': ccrs.Robinson()}
    )
    
    # Define common plotting configurations
    map_features = [
        (cfeature.COASTLINE, 1.0, 'black'),
        (cfeature.BORDERS, 0.5, 'gray')
    ]
    
    ax_win = axes[0]
    for feature, lw, color in map_features:
        ax_win.add_feature(feature, linewidth=lw, edgecolor=color)
    ax_win.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    
    im_win = winter_source.plot(
        ax=ax_win, 
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r', 
        center=0, 
        # robust=True,
        # vmin=-1, vmax=1,  # Set fixed color limits for better comparison
        vmin=-0.5, vmax=0.5,  # Set fixed color limits for better comparison
        cbar_kwargs={'label': 'Anomaly (ppm)', 'orientation': 'horizontal', 'pad': 0.05, 'shrink': 0.8}
    )
    ax_win.set_title(f"DJF Magnitude ({year_to_plot})", fontsize=12, fontweight='bold')
    gl_win = ax_win.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl_win.top_labels, gl_win.right_labels = False, False

    ax_sum = axes[1]
    for feature, lw, color in map_features:
        ax_sum.add_feature(feature, linewidth=lw, edgecolor=color)
    ax_sum.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    
    im_sum = summer_sink.plot(
        ax=ax_sum, 
        transform=ccrs.PlateCarree(),
        cmap='PiYG_r', 
        center=0, 
        # robust=True,
        vmin=-0.5, vmax=0.5,  # Set fixed color limits for better comparison
        cbar_kwargs={'label': 'Anomaly (ppm)', 'orientation': 'horizontal', 'pad': 0.05, 'shrink': 0.8}
    )
    ax_sum.set_title(f"JJA Magnitude ({year_to_plot})", fontsize=12, fontweight='bold')
    gl_sum = ax_sum.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl_sum.top_labels, gl_sum.left_labels = False, False  # Hide left labels on the second plot to avoid overlap

    # Global adjustments
    plt.suptitle(f"Carbon Footprint Anomalies (Relative to Yearly Global Mean)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_clean_global_flux(data, year_to_plot=1900):
    """
    Plots seamless global maps of DJF vs JJA net surface CO2 fluxes
    using a single unified variable and colorbar scale.
    """
    # 1. Filter for the year
    yearly_flux = data["co2_flx_total_inst"].sel(time=str(year_to_plot))

    # 2. Extract seasonal averages
    winter_months = yearly_flux['time.month'].isin([12, 1, 2])
    winter_flux = yearly_flux.where(winter_months, drop=True).mean(dim='time')

    summer_months = yearly_flux['time.month'].isin([6, 7, 8])
    summer_flux = yearly_flux.where(summer_months, drop=True).mean(dim='time')

    # 3. Create the 1x2 map grid
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), subplot_kw={'projection': ccrs.Robinson()})
    map_features = [(cfeature.COASTLINE, 0.8, 'black'), (cfeature.BORDERS, 0.4, 'gray')]
    
    # Unified scale to keep land and ocean physically comparable
    v_limit = 1.0e-7 
    cbar_labels = 'Flux Rate (kg m$^{-2}$ s$^{-1}$)'

    # --- Left Plot: Winter (DJF) ---
    ax_win = axes[0]
    for feature, lw, color in map_features:
        ax_win.add_feature(feature, linewidth=lw, edgecolor=color)
    
    im_win = winter_flux.plot(
        ax=ax_win, transform=ccrs.PlateCarree(), cmap='RdBu_r', center=0,
        vmin=-v_limit, vmax=v_limit,
        cbar_kwargs={'label': cbar_labels, 'orientation': 'horizontal', 'pad': 0.05, 'shrink': 0.8}
    )
    ax_win.set_title(f"DJF Net Surface Flux ({year_to_plot})", fontsize=12, fontweight='bold')

    # --- Right Plot: Summer (JJA) ---
    ax_sum = axes[1]
    for feature, lw, color in map_features:
        ax_sum.add_feature(feature, linewidth=lw, edgecolor=color)
        
    im_sum = summer_flux.plot(
        ax=ax_sum, transform=ccrs.PlateCarree(), cmap='RdBu_r', center=0,
        vmin=-v_limit, vmax=v_limit,
        cbar_kwargs={'label': cbar_labels, 'orientation': 'horizontal', 'pad': 0.05, 'shrink': 0.8}
    )
    ax_sum.set_title(f"JJA Net Surface Flux ({year_to_plot})", fontsize=12, fontweight='bold')

    # Global adjustments
    plt.suptitle(f"flat10 Coupled Carbon Fluxes (Positive = Source, Negative = Sink)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


# def plot_co2_flux_vs_year(data, year_to_plot=1900):
#     """
#     Plots side-by-side geographic maps showing the winter (DJF) and 
#     summer (JJA) surface CO2 flux rates for a given year, splitting
#     land and ocean onto optimized color scales.
#     """
#     # 1. Select total instantaneous flux and filter for the year
#     yearly_flux = data["co2_flx_total_inst"].sel(time=str(year_to_plot))
    
#     # Identify Land vs Ocean Mask
#     # ECHAM6 natively includes a Sea-Land Mask ('slm': 1=land, 0=ocean). 
#     # If not present, we create a proxy mask where variance is high (land features high seasonal variance)
#     if 'slm' in data:
#         is_land = data['slm'] >= 1
#     else:
#         is_land = yearly_flux.std(dim='time') > 1e-9

#     # 2. Extract seasonal averages
#     winter_months = yearly_flux['time.month'].isin([12, 1, 2])
#     winter_flux = yearly_flux.where(winter_months, drop=True).mean(dim='time')

#     summer_months = yearly_flux['time.month'].isin([6, 7, 8])
#     summer_flux = yearly_flux.where(summer_months, drop=True).mean(dim='time')

#     # 3. Set up the figure using a Map Projection
#     fig, axes = plt.subplots(
#         1, 2, 
#         figsize=(18, 7.5),  # Slightly increased height to accommodate stacked colorbars
#         subplot_kw={'projection': ccrs.Robinson()}
#     )
    
#     map_features = [
#         (cfeature.COASTLINE, 0.8, 'black'),
#         (cfeature.BORDERS, 0.4, 'gray')
#     ]
    
#     # Set explicit, fixed physical scales for flat10 data boundaries
#     land_vmin, land_vmax = -1.0e-7, 1.0e-7
#     ocean_vmin, ocean_vmax = -3.0e-9, 3.0e-9
    
#     # ----------------------------------------------------
#     # --- Left Plot: Winter (DJF) Land + Sea Split ---
#     # ----------------------------------------------------
#     ax_win = axes[0]
#     for feature, lw, color in map_features:
#         ax_win.add_feature(feature, linewidth=lw, edgecolor=color)
        
#     # Layer 1: Ocean Flux (Highly Sensitive Scale)
#     im_win_ocean = winter_flux.where(~is_land).plot(
#         ax=ax_win, transform=ccrs.PlateCarree(),
#         cmap='PuOr', center=0, robust=False, vmin=ocean_vmin, vmax=ocean_vmax, add_colorbar=False
#     )
#     # Layer 2: Land Flux (Wide Scale) overlaid on top
#     im_win_land = winter_flux.where(is_land).plot(
#         ax=ax_win, transform=ccrs.PlateCarree(),
#         cmap='RdBu_r', center=0, robust=False, vmin=land_vmin, vmax=land_vmax, add_colorbar=False
#     )
    
#     ax_win.set_title(f"DJF Net Surface Flux ({year_to_plot})", fontsize=12, fontweight='bold')
#     gl_win = ax_win.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
#     gl_win.top_labels, gl_win.right_labels = False, False

#     # Add separate stacked colorbars for DJF
#     cbar_win_l = fig.colorbar(im_win_land, ax=ax_win, orientation='horizontal', pad=0.06, shrink=0.7)
#     cbar_win_l.set_label('Land Flux (kg m$^{-2}$ s$^{-1}$)', fontsize=9)
#     cbar_win_o = fig.colorbar(im_win_ocean, ax=ax_win, orientation='horizontal', pad=0.06, shrink=0.7)
#     cbar_win_o.set_label('Ocean Flux (kg m$^{-2}$ s$^{-1}$)', fontsize=9)

#     # ----------------------------------------------------
#     # --- Right Plot: Summer (JJA) Land + Sea Split ---
#     # ----------------------------------------------------
#     ax_sum = axes[1]
#     for feature, lw, color in map_features:
#         ax_sum.add_feature(feature, linewidth=lw, edgecolor=color)
        
#     # Layer 1: Ocean Flux
#     im_sum_ocean = summer_flux.where(~is_land).plot(
#         ax=ax_sum, transform=ccrs.PlateCarree(),
#         cmap='PuOr', center=0, robust=False, vmin=ocean_vmin, vmax=ocean_vmax, add_colorbar=False
#     )
#     # Layer 2: Land Flux
#     im_sum_land = summer_flux.where(is_land).plot(
#         ax=ax_sum, transform=ccrs.PlateCarree(),
#         cmap='RdBu_r', center=0, robust=False, vmin=land_vmin, vmax=land_vmax, add_colorbar=False
#     )
    
#     ax_sum.set_title(f"JJA Net Surface Flux ({year_to_plot})", fontsize=12, fontweight='bold')
#     gl_sum = ax_sum.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
#     gl_sum.top_labels, gl_sum.left_labels = False, False

#     # Add separate stacked colorbars for JJA
#     cbar_sum_l = fig.colorbar(im_sum_land, ax=ax_sum, orientation='horizontal', pad=0.06, shrink=0.7)
#     cbar_sum_l.set_label('Land Flux (kg m$^{-2}$ s$^{-1}$)', fontsize=9)
#     cbar_sum_o = fig.colorbar(im_sum_ocean, ax=ax_sum, orientation='horizontal', pad=0.06, shrink=0.7)
#     cbar_sum_o.set_label('Ocean Flux (kg m$^{-2}$ s$^{-1}$)', fontsize=9)

#     # Global adjustments
#     plt.suptitle(f"flat10 Coupled Flux Footprints (Positive = Source, Negative = Sink)", fontsize=15, fontweight='bold', y=0.98)
#     plt.show()

def plot_land_vs_ocean_flux(data, year_to_plot=1900, season='JJA'):
    """
    Plots side-by-side maps comparing Ocean Surface Flux vs Land Surface Flux
    for a chosen year and season (e.g., 'JJA' or 'DJF').
    """
    # 1. Filter data for the chosen year
    yearly_data = data.sel(time=str(year_to_plot))
    
    # 2. Define the seasonal months
    if season == 'DJF':
        months = [12, 1, 2]
    elif season == 'JJA':
        months = [6, 7, 8]
    else:
        raise ValueError("Choose 'DJF' or 'JJA'")
        
    # 3. Extract the two distinct variables averaged over the season
    ocean_flux = yearly_data["co2_flx_ocean"].where(yearly_data['time.month'].isin(months), drop=True).mean(dim='time')
    
    # Using the dedicated land flux variable from your file metadata
    land_flux = yearly_data["co2_flx_inst"].where(yearly_data['time.month'].isin(months), drop=True).mean(dim='time')

    # 4. Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), subplot_kw={'projection': ccrs.Robinson()})
    map_features = [(cfeature.COASTLINE, 1.0, 'black'), (cfeature.BORDERS, 0.5, 'gray')]
    
    # Unified scale works beautifully since both are highly active in flat10
    land_limit = 2.0e-7 
    ocean_limit = 8.0e-9
    cbar_labels = 'Flux Rate (kg m$^{-2}$ s$^{-1}$)'

    # --- Left Plot: Ocean Flux ---
    ax_ocn = axes[0]
    for feature, lw, color in map_features:
        ax_ocn.add_feature(feature, linewidth=lw, edgecolor=color)
    
    ocean_flux.plot(
        ax=ax_ocn, transform=ccrs.PlateCarree(), cmap='RdBu_r', center=0,
        vmin=-ocean_limit, vmax=ocean_limit,
        cbar_kwargs={'label': cbar_labels, 'orientation': 'horizontal', 'pad': 0.05, 'shrink': 0.8}
    )
    ax_ocn.set_title(f"{season} Ocean Surface CO$_2$ Flux ({year_to_plot})", fontsize=12, fontweight='bold')

    # --- Right Plot: Land Flux ---
    ax_lnd = axes[1]
    for feature, lw, color in map_features:
        ax_lnd.add_feature(feature, linewidth=lw, edgecolor=color)
        
    land_flux.plot(
        ax=ax_lnd, transform=ccrs.PlateCarree(), cmap='PRGn', center=0, # Using green/purple for land vegetation
        vmin=-land_limit, vmax=land_limit,
        cbar_kwargs={'label': cbar_labels, 'orientation': 'horizontal', 'pad': 0.05, 'shrink': 0.8}
    )
    ax_lnd.set_title(f"{season} Land Surface CO$_2$ Flux ({year_to_plot})", fontsize=12, fontweight='bold')

    plt.suptitle(f"flat10 Carbon Exchange Fluxes (Positive = Source, Negative = Sink)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("path", help="path to data file")
    # parser.add_argument("year", help="year of data file")
    # parser.add_argument("plev", type=int, help="pressure level")
    parser.add_argument("--zec", action="store_true", help="use zec data")
    parser.add_argument("--flux", action="store_true", help="use flux data")


    args = parser.parse_args()
    log_args(args)

    # 1. Initialize variables and tracking arrays
    years = np.arange(1850, 1950, 10, dtype=int)  
    layers = np.arange(0, 2, 1, dtype=int) 

    # 2. Determine the correct file path BEFORE opening the dataset
    if args.zec:
        path = '/media/lisa/PortableSSD/Data/flat10/HW312_esm-flat10-zec_RESCUE/echam6/HW312_esm-flat10-zec_RESCUE_echam6_tracer_1850-1949.nc'
    elif args.flux:
        path = '/media/lisa/PortableSSD/Data/flat10/HW310_esm-flat10_RESCUE/echam6/HW310_esm-flat10_RESCUE_echam6_co2_1850-1949.nc'
    else:
        # Default fallback path
        path = '/media/lisa/PortableSSD/Data/flat10/HW310_esm-flat10_RESCUE/echam6/HW310_esm-flat10_RESCUE_echam6_tracer_1850-1949.nc'

    print(f"Loading dataset from: {path}")

    # 3. Open dataset securely
    data = xr.open_dataset(path, engine="netcdf4")

    # 4. Handle vertical level collapsing if 'lev' coordinate is present in the file
    if 'lev' in data.coords or 'lev' in data.data_vars:
        print(f"Collapsing vertical layers: {layers}")
        data = data.sel(lev=slice(layers[0], layers[-1])).mean(dim='lev')

    # 5. Route dataset to the correct plotting routine
    print(f"Generating maps for years: {years}")

    for year in years:
        if args.flux:
            # Use the specific surface flux script (physical 0-baseline)
            plot_land_vs_ocean_flux(data, year_to_plot=year, season='JJA')
            plot_land_vs_ocean_flux(data, year_to_plot=year, season='DJF')
            # plot_clean_global_flux(data, year_to_plot=year)
        else:
            # Use the concentration script (global mean anomalies subtracted)
            plot_co2_anomalies_vs_year(data, year_to_plot=year)


