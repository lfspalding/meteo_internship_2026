import calendar
import matplotlib
matplotlib.use('qtagg')  # Use non-interactive backend
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import argparse
import numpy as np

from helpers import convert_ppm, get_month, log_args, wrap_data


def anomaly_plot(path, years, months, plevs, rows=2, zec=False):

    data = xr.load_dataset(path, engine="netcdf4")
    
    for plev in plevs:
        plev_data = data.sel(plev=plev, time=slice(get_month(years[0], 1), get_month(years[-1], 12)))
        
        for month in months:
            fig, axes = plt.subplots(nrows=rows, ncols=(len(years)+(rows-1))//rows, figsize=(18, 8),
                                     subplot_kw={'projection': ccrs.Robinson()})
            axes = axes.flatten()
            
            month_data = plev_data["CO2"].where(plev_data.time.dt.month == month, drop=True)
            global_mean = month_data.mean(dim=["lat", "lon"])
            # global_trend_line = global_mean - detrend(global_mean, type="linear")

            poly_coefficients = global_mean.polyfit(dim="time", deg=1)
            # global_trend_line = xr.polyval(global_mean.time, poly_coefficients.CO2_polyfit_coefficients)
            # Change Line 30 to this:
            global_trend_line = xr.polyval(global_mean.time, poly_coefficients["polyfit_coefficients"])

            detrended_month_data = month_data - global_trend_line

            # xr.apply_ufunc(
            #     detrend,
            #     da,
            #     kwargs={"axis": 0, "type": "linear"},
            #     input_core_dims=[["time"]],
            #     output_core_dims=[["time"]],
            # )
            
            detrended_ppm = convert_ppm(detrended_month_data, key=False)
            wrapped_detrended = wrap_data(detrended_ppm)
            
            # Calculates a shared max value across all years so subplots share the exact same scale
            max_val = float(np.abs(wrapped_detrended).max())
            v_limit = max_val if max_val > 0 else 1e-11 # Fallback safeguard if data is completely flat
           
            for i, year in enumerate(years):
                # Pull the specific year's detrended state using time accessors
                # This avoids any KeyError mismatch risks with timestamp string hours/minutes
                now_detrended = wrapped_detrended.where(wrapped_detrended.time.dt.year == year, drop=True)
                
                # Plot
                ax = axes[i]
                ax.set_global()
                ax.coastlines()
                
                cbar_kwargs = {
                    'orientation': 'horizontal', 
                    'shrink': 1, 
                    'pad': .05, 
                    'aspect': 60, 
                    'label': f'CO2 anomaly {now_detrended.attrs.get("units", "unknown")}, {year}'
                }
                
                contour_plot = now_detrended.squeeze("time").plot.contourf(
                    ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs, 
                    levels=29, cmap="coolwarm", vmin=-v_limit, vmax=v_limit
                )
                ax.set_title(f"CO2 Detrended Anomaly for {year}-{month:02d}")

            # Hide any remaining unused subplots if your year count is odd
            for j in range(len(years), len(axes)):
                axes[j].axis('off')

            plt.suptitle(f'CO2 Levels for {calendar.month_name[month]}, Plev {plev/100} hPa', fontsize=16)
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
    years = np.arange(1920, 1946, 5, dtype=int)
    print(years)
    months = [9] # np.arange(1, 13, dtype=int)
    print(months)
    plevs = [10000]
    print(plevs)
    anomaly_plot(path, years, months, plevs, rows=2, zec=args.zec)
    