import calendar
import matplotlib
matplotlib.use('qtagg')  # Use non-interactive backend
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import argparse
import numpy as np

from helpers import convert_ppm, get_month, log_args, wrap_data


# def map_cycle(path, years, months, plevs, rows=2, zec=False):

#     data = xr.load_dataset(path, engine="netcdf4")
    
#     for plev in plevs:
#         plev_data = data.sel(plev=plev)
#         # TODO slice time to the range of years provided, if needed
#         all_time_mean = plev_data["CO2"].mean(dim="time")

#         fig, axes = plt.subplots(nrows=rows, ncols=(len(months)+(rows-1))//rows, figsize=(16, 12),
#                                      subplot_kw={'projection': ccrs.Robinson()})
#         axes = axes.flatten()
        
#         for i, month in enumerate(months):
#             monthly_mean = plev_data["CO2"].where(plev_data.time.dt.month == month, drop=True).mean(dim="time")
#             monthly_anomaly = monthly_mean - all_time_mean
#             get_ppm = convert_ppm(monthly_anomaly, key=False)
#             wrapped = wrap_data(get_ppm)
            
#             # Plot
#             ax = axes[i]
#             ax.set_global()
#             ax.coastlines()
            
#             cbar_kwargs = {
#                 'orientation': 'horizontal', 
#                 'shrink': 1, 
#                 'pad': .05, 
#                 'aspect': 60, 
#                 'label': f'CO2 anomaly {wrapped.attrs.get("units", "unknown")}, {month:02d}'
#             }
            
#             contour_plot = wrapped.plot.contourf(
#                 ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs, 
#                 levels=29, cmap="viridis", vmin=-1.0e-11, vmax=1.0e-11
#             ) # 
#             ax.set_title(f"CO2 Anomaly for {calendar.month_name[month]}")

#         # Hide any remaining unused subplots if your year count is odd
#         for j in range(len(months), len(axes)):
#             axes[j].axis('off')

#         plt.suptitle(f'Seasonal CO2 Levels for Plev {plev/100} hPa', fontsize=16)
#         plt.tight_layout()
#         plt.show()

def map_cycle(path, years, months, plevs, rows=2, zec=False):

    data = xr.load_dataset(path, engine="netcdf4")
    trend_coefficients = data["CO2"].polyfit(dim="time", deg=1)
    
    # Evaluate the regression line at every time step
    fitted_trend = xr.polyval(data["time"], trend_coefficients.polyfit_coefficients)
    
    # Subtract the trend to yield a flat, stationary time series
    detrended_CO2 = data["CO2"] - fitted_trend
    
    # 1. Vectorized processing: Calculate all-time mean and monthly means for all plevs at once
    # TODO slice time to the range of years provided, if needed
    all_time_mean = detrended_CO2.mean(dim="time")
    
    # Group by month and average, then filter only the requested months
    monthly_means = detrended_CO2.groupby("time.month").mean(dim="time")
    
    # Calculate anomalies (xarray automatically broadcasts all_time_mean across the month dimension)
    all_anomalies = monthly_means - all_time_mean
    
    # Apply your custom processing functions to the entire xarray object at once
    all_get_ppm = convert_ppm(all_anomalies, key=False)
    all_wrapped = wrap_data(all_get_ppm)

    # 2. Loop only over pressure levels for plotting
    for plev in plevs:
        # Extract the pre-computed wrapped data for this specific plev
        plev_wrapped = all_wrapped.sel(plev=plev)
        
        # Calculate dynamic, symmetric min/max limits instantly from the array
        max_abs = float(abs(plev_wrapped).max())
        dynamic_vmin = -max_abs
        dynamic_vmax = max_abs

        fig, axes = plt.subplots(nrows=rows, ncols=(len(months)+(rows-1))//rows, figsize=(18, 13),
                                 subplot_kw={'projection': ccrs.Robinson()})
        axes = axes.flatten()
        
        # Enumerate over the months dimension of our pre-computed xarray object
        for i, month in enumerate(months):
            # Extract single month data without recomputing anything
            wrapped_month = plev_wrapped.sel(month=month)
            
            # Plot
            ax = axes[i]
            ax.set_global()
            ax.coastlines()
            
            cbar_kwargs = {
                'orientation': 'horizontal', 
                'shrink': 1, 
                'pad': .05, 
                'aspect': 60, 
                'label': f'CO2 anomaly {wrapped_month.attrs.get("units", "unknown")}, {month:02d}'
            }
            
            contour_plot = wrapped_month.plot.contourf(
                ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs, 
                levels=29, cmap="viridis", vmin=dynamic_vmin, vmax=dynamic_vmax
            ) 
            ax.set_title(f"CO2 Anomaly for {calendar.month_name[month]}")

        # Hide any remaining unused subplots if your year count is odd
        for j in range(len(months), len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'Detrended Seasonal CO2 Levels for Plev {plev/100} hPa', fontsize=16)
        plt.tight_layout()
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
        print("NOT SET UP FOR ZEC DATA YET")
        # path = '/media/lisa/PortableSSD/Data/flat10/HW312_esm-flat10-zec_RESCUE/echam6/'
    print(path)
    years = np.arange(1920, 1930, dtype=int)
    # print(years)
    months = np.arange(1, 13, dtype=int)
    print(months)
    plevs = [100000, 77500, 50000, 30000, 10000, 5000, 1000, 500, 50]
    print(plevs)
    map_cycle(path, years, months, plevs, rows=3, zec=args.zec)
    