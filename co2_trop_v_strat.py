import argparse
import xarray as xr
import numpy as np
from helpers import log_args, convert_ppm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# def co2_vs_time(data, years):
#     lat1, lat2 = 5, -20
#     lon1, lon2 = 280, 310

#     data = data.sel(
#         lat=slice(lat1, lat2),
#         lon=slice(lon1, lon2),
#         time=slice(f"{years[0]}-01-01", f"{years[-1]}-12-31"),
#     )
#     data = data.sel(plev=slice(100000, 1000))
#     data = data.mean(dim=["lat", "lon"])
#     data_ppm = convert_ppm(data)
#     data_ppm["plev"] = data_ppm["plev"] / 100
#     data_ppm["CO2"].transpose("plev", "time").plot.contourf(
#         x="time", y="plev", cmap="viridis", levels=20
#     )
#     plt.gca().invert_yaxis()

#     plt.title("CO2 Vertical Profile Over Time")
#     plt.ylabel("Pressure Level (hPa)")
#     plt.xlabel("Time")
#     plt.title(
#         f"CO2 Concentration Over Time, lat {lat1} to {lat2}, lon {lon1} to"
#         f" {lon2}, years {years[0]}-{years[-1]}"
#     )
#     plt.yscale("log")
#     plt.show()


# def plot_co2_troposphere_vs_stratosphere(data, years, zec=False):
#     """
#     Compares global mean CO2 concentrations over time between the Troposphere 
#     (Surface to 200 hPa) and Stratosphere (200 hPa to 10 hPa).
#     """
#     # 1. Collapse spatial dimensions to create a clean time/height profile
#     global_profile = data["CO2"].mean(dim=['lat', 'lon'])
    
#     # 2. Match pressure levels in Pascals (1 hPa = 100 Pa)
#     # Troposphere: 1000 hPa down to 200 hPa -> 100,000 Pa to 20,000 Pa
#     # Stratosphere: 200 hPa down to 10 hPa   -> 20,000 Pa to 1,000 Pa
#     # Note: Using slice(None) or min/max bounds since pressure axis orientation can vary (up or down)
#     p_levels = global_profile['plev'].values
    
#     trop_mask = (p_levels <= 100000) & (p_levels >= 20000)
#     strat_mask = (p_levels < 20000) & (p_levels >= 1000)
    
#     # 3. Calculate layer averages weighted by plev depth if needed
#     troposphere_mean_raw = global_profile.isel(plev=trop_mask).mean(dim='plev')
#     stratosphere_mean_raw = global_profile.isel(plev=strat_mask).mean(dim='plev')
    
#     # Convert arrays to ppm unit values
#     troposphere_mean = convert_ppm(troposphere_mean_raw, key=None)
#     stratosphere_mean = convert_ppm(stratosphere_mean_raw, key=None)

#     # 4. Extract time vectors
#     years = troposphere_mean.time.dt.year.values if hasattr(troposphere_mean.time, 'dt') else troposphere_mean.time.values

#     # 5. Plotting Time Trends
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     ax.plot(years, troposphere_mean.values, color='#1f77b4', linewidth=2.5, label='📊 Troposphere Average (Surface - 200 hPa)')
#     ax.plot(years, stratosphere_mean.values, color='#ff7f0e', linewidth=2.5, linestyle='--', label='🚀 Stratosphere Average (200 - 10 hPa)')
    
#     # Formatting
#     ax.set_title(f"Global CO2 Accumulation: Troposphere vs. Stratosphere", fontsize=14, fontweight='bold', pad=15)
#     ax.set_ylabel("Atmospheric CO2 Concentration (ppm)", fontsize=12)
#     ax.set_xlabel("Model Year", fontsize=12)
#     ax.grid(True, alpha=0.3, linestyle='--')
#     ax.legend(loc="upper left", fontsize=11)
    
#     fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)
#     plt.show()


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
    ax.set_ylabel("Atmospheric CO2 Concentration (ppm)", fontsize=12)
    ax.set_xlabel("Model Year", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc="upper left", fontsize=11)
    
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("path", help="path to data file")
    # parser.add_argument("year", help="year of data file")
    # parser.add_argument("plev", type=int, help="pressure level")
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_args(args)

    path = '/media/lisa/PortableSSD/Data/flat10/HW310_esm-flat10_RESCUE/echam6/HW310_esm-flat10_RESCUE_echam6_tracerpl_1850-1949.nc'
    if (args.zec):
        path = '/media/lisa/PortableSSD/Data/flat10/HW312_esm-flat10-zec_RESCUE/echam6/HW312_esm-flat10-zec_RESCUE_echam6_tracerpl_1850-1949.nc'
    print(path)
    years = np.arange(1900, 1950, 1, dtype=int)
    print(years)
    months = [9] # np.arange(1, 13, dtype=int)
    print(months)

    data = xr.open_dataset(path)
    data = data.sel(time=slice(f"{years[0]}-01-01", f"{years[-1]}-12-31"))

    plot_co2_monthly_trend(data, years, zec=args.zec)
