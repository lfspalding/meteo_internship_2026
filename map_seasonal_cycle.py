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

    # ## METHOD 1
    # trend_coefficients = data["CO2"].polyfit(dim="time", deg=1)

    # # Evaluate the regression line at every time step
    # fitted_trend = xr.polyval(data["time"], trend_coefficients.polyfit_coefficients)

    # # Subtract the trend to yield a flat, stationary time series
    # detrended_CO2 = data["CO2"] - fitted_trend

    # # Group by month and average, then filter only the requested months
    # monthly_means = detrended_CO2.groupby("time.month").mean(dim="time")

    ## METHOD 2
    # Group by month and calculate the mean over the time dimension
    # monthly_means = data["CO2"].groupby("time.month").mean(dim="time")

    # # Fit the polynomial trend on the monthly averaged data
    # trend_coefficients = monthly_means.polyfit(dim="month", deg=1)

    # # Evaluate the regression line across the month dimension
    # fitted_trend = xr.polyval(monthly_means["month"], trend_coefficients.polyfit_coefficients)

    # # Subtract the trend from the monthly means to get detrended anomalies
    # monthly_means = monthly_means - fitted_trend

    ## METHOD 3
    # # 1. Group by month and average over time first
    # monthly_means = data["CO2"].groupby("time.month").mean(dim="time")

    # # 2. Calculate the global average for each month (mean across spatial dimensions)
    # global_monthly_mean = monthly_means.mean(dim=["lat", "lon"])

    # # 3. Fit the linear trend to the global average monthly data
    # trend_coefficients = global_monthly_mean.polyfit(dim="month", deg=1)

    # # 4. Evaluate the global regression line across the month dimension
    # fitted_global_trend = xr.polyval(monthly_means["month"], trend_coefficients.polyfit_coefficients)

    # # 5. Subtract the global trend from the spatially varying monthly data
    # # xarray automatically broadcasts the 1D trend across spatial dimensions
    # monthly_means = monthly_means - fitted_global_trend

    ## METHOD 4
    # global_time_series = data["CO2"].mean(dim=["lat", "lon"])

    # # 2. Fit the linear trend to the full global average time series
    # trend_coefficients = global_time_series.polyfit(dim="time", deg=1)

    # # 3. Evaluate the global regression line at every original time step
    # fitted_global_trend = xr.polyval(data["time"], trend_coefficients.polyfit_coefficients)

    # # 4. Subtract the global trend from the raw, spatially varying data
    # detrended_data = data["CO2"] - fitted_global_trend

    # # 5. Group the detrended data by month and average over time
    # monthly_means = detrended_data.groupby("time.month").mean(dim="time")

    ## METHOD 5
    # 1. Group by month to isolate each calendar month across all years
    # monthly_groups = data["CO2"].groupby("time.month")

    # # 2. Fit a linear trend for each individual month across the time dimension
    # # This returns trend coefficients for each month (dimension: month, degree)
    # monthly_trend_coefficients = monthly_groups.map(lambda x: x.polyfit(dim="time", deg=1))
    
    # # 3. Evaluate the month-specific trends at every original time step
    # # This forces xr.polyval to evaluate each month independently on its own slice
    # fitted_monthly_trend = monthly_groups.map(
    #     lambda x: xr.polyval(x["time"], monthly_trend_coefficients.sel(month=x["time.month"][0]).polyfit_coefficients)
    # )
    # # 4. Subtract the month-specific trends from the raw data to detrend it
    # detrended_data = data["CO2"] - fitted_monthly_trend
    # monthly_means = detrended_data.groupby("time.month").mean(dim=["time"])

    # # 5. Extract the trend strength (the slope/rate of change) for each month
    # # The slope is the coefficient for degree 1
    # monthly_slopes = monthly_trend_coefficients.polyfit_coefficients.sel(degree=1)

    ## METHOD 6
    # 1. Group by month and calculate the spatial coefficients
    monthly_groups = data["CO2"].groupby("time.month")
    monthly_trend_coefficients = monthly_groups.map(lambda x: x.polyfit(dim="time", deg=1))

    # 2. Average the coefficients globally (collapsing lat and lon)
    # This extracts the global trend for each specific month
    global_monthly_coefficients = monthly_trend_coefficients.mean(dim=["lat", "lon"])

    # 3. Evaluate the global trend line piece-by-piece to save memory
    fitted_global_monthly_trend = monthly_groups.map(
        lambda x: xr.polyval(x["time"], global_monthly_coefficients.sel(month=x["time.month"]).polyfit_coefficients)
    )

    # 4. Subtract the global monthly trend from your raw, spatially varying data
    detrended_data = data["CO2"] - fitted_global_monthly_trend

    # 5. Generate your spatial maps from the globally detrended data
    monthly_means = detrended_data.groupby("time.month").mean(dim="time")

    # 6. Extract the global trend strength per month (1D array of 12 values)
    monthly_slopes = global_monthly_coefficients.polyfit_coefficients.sel(degree=1)

    # Apply your custom processing functions to the entire xarray object at once
    all_get_ppm = convert_ppm(monthly_means, key=False)
    all_wrapped = wrap_data(all_get_ppm)
    ns_per_year = 365.25 * 24 * 60 * 60 * 1e9
    monthly_slopes = convert_ppm(monthly_slopes, key=False) * ns_per_year


    # 2. Loop only over pressure levels for plotting
    for plev in plevs:
        # Extract the pre-computed wrapped data for this specific plev
        plev_wrapped = all_wrapped.sel(plev=plev) # * 1e10
        
        # Calculate dynamic, symmetric min/max limits instantly from the array
        max_abs = float(abs(plev_wrapped).max())
        dynamic_vmin = float(plev_wrapped.min())
        dynamic_vmax = float(plev_wrapped.max())

        fig, axes = plt.subplots(nrows=rows, ncols=(len(months)+(rows-1))//rows, figsize=(18, 13),
                                 subplot_kw={'projection': ccrs.Robinson()})
        axes = axes.flatten()
        
        # Enumerate over the months dimension of our pre-computed xarray object
        for i, month in enumerate(months):
            print(monthly_slopes.sel(plev=plev, month=i+1))
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
                'label': f'CO2 ppm, Trend Slope {monthly_slopes.sel(plev=plev, month=i+1).values:.4e}',
                # 'format': '%.2f' (x 10⁻¹⁰)
            }
            
            contour_plot = wrapped_month.plot.contourf(
                ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs, 
                levels=29, cmap="viridis", vmin=dynamic_vmin, vmax=dynamic_vmax
            ) 
            ax.set_title(f"{calendar.month_name[month]}")

        # Hide any remaining unused subplots if your year count is odd
        for j in range(len(months), len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'Seasonal CO2 Levels for Plev {plev/100} hPa, Monthly Mean Trend Removed', fontsize=16)
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
    plevs = [50000]
    print(plevs)
    map_cycle(path, years, months, plevs, rows=3, zec=args.zec)
    