from helpers import log_args, remove_seasonal_cycle
import numpy as np
from scipy import signal
import xarray as xr
import matplotlib.pyplot as plt
import argparse


def ccf_values(series1, series2):
    p = series1
    q = series2
    p = (p - np.mean(p)) / (np.std(p) * len(p))
    q = (q - np.mean(q)) / (np.std(q))  
    c = np.correlate(p, q, 'full')
    return c


def create_ccf_plot(lags, ccf, data_len, ax, ylim, color):
    ax.plot(lags, ccf, color=color, label="correlation")
    ax.axhline(-1.96/np.sqrt(data_len), color='red', label='95% confidence interval')
    ax.axhline(1.96/np.sqrt(data_len), color='red')
    ax.axvline(x = 0, color = 'black', lw = 1)
    ax.axhline(y = 0, color = 'black', lw = 1)
    ax.axhline(y = np.max(ccf), color = 'blue', lw = 1, 
    linestyle='--', label = 'highest +/- correlation')
    ax.axhline(y = np.min(ccf), color = 'blue', lw = 1, 
    linestyle='--')
    # ax.set(ylim = ylim)
    ax.set_ylabel('Correlation Coefficients', weight='bold', fontsize = 12)
    ax.set_xlabel('Time Lag (Months)', weight='bold', fontsize = 12)
    # ax.set_xlim(-max-min, max-min)
    ax.legend()


def just_math_ccf(path, plevs):
    data = xr.load_dataset(path, engine="netcdf4")
    start_time = '1900-01-31T23:52:00.000000000'
    end_time = '1930-12-31T23:52:00.000000000'
    data = data.sel(time=slice(start_time, end_time))
    for plev in plevs:
        lags_plev, ccf_plev, data_len, base = do_calc(data, plev)
        max_idx = np.argmax(ccf_plev)
        exact_lag = lags_plev[max_idx]
        max_corr = ccf_plev[max_idx]
        print(f"{plev/100} hPa: Max corr {max_corr:.3f} at {exact_lag} Months")


def do_calc(data, plev):
    # compare to baseline
    base = 9.25e+04
    
    # 1. Extract the raw slices
    baseline = data.sel(plev=base, lat=slice(30, -30)).mean(dim=["lon", "lat"])
    plev_data = data.sel(plev=plev, lat=slice(30, -30)).mean(dim=["lon", "lat"])

    # 2. Apply a 12-month rolling mean to completely smooth out the seasonal cycle
    baseline_smooth = baseline["CO2"].rolling(time=12, center=True).mean().dropna(dim="time")
    plev_smooth = plev_data["CO2"].rolling(time=12, center=True).mean().dropna(dim="time")

    # 3. Align the time coordinates after dropping NaN edges
    baseline_smooth, plev_smooth = xr.align(baseline_smooth, plev_smooth)

    # 4. Difference the smoothed data to kill the massive global trend
    baseline_final = baseline_smooth.diff(dim="time")
    plev_final = plev_smooth.diff(dim="time")

    # 5. Extract the clean arrays for the CCF function
    y1_prime = plev_final.values
    y2_prime = baseline_final.values

    # 6. Recalculate lags based on the new length
    data_len = len(y1_prime)
    lags_plev = signal.correlation_lags(data_len, len(y2_prime))
    ccf_plev = ccf_values(y1_prime, y2_prime)

    return lags_plev, ccf_plev, data_len, base


def ccf_table(path, base_plev, start, end):
    # 1. Fetch your Xarray DataArray and convert Pa to hPa
    da_lag = ccf_table_to_xarray(path, base_plev, start, end)
    da_lag = da_lag.assign_coords(plev=da_lag.plev / 100.0)

    # 2. Setup a standard figure canvas
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.rcParams.update({"font.size": 12})

    # 3. Clean, direct 2D plot call using built-in Xarray wrappers
    da_lag.plot(
        ax=ax,
        x="lat",
        y="plev",
        cmap="RdBu_r",
        robust=True,
        cbar_kwargs={"label": "Lag (Months)"},  # Automatic inline colorbar setup
    )

    # 1. Apply Logarithmic Scale to the Pressure Axis
    ax.set_yscale("log")

    # 2. Get structural axis edges from your data limits
    y_start = float(da_lag.plev.values[0])   # e.g., ~925 hPa
    y_end = float(da_lag.plev.values[-1])   # e.g., ~0.1 hPa or 1 hPa
    ax.set_ylim(y_start, y_end)

    # 3. CRUCIAL FIX: Define clean, standard atmospheric landmarks for log scale
    # This completely overrides the scientific 10^0 notation
    custom_ticks = [0.1, 0.5, 1, 5, 10, 20, 50, 100, 300, 500, 700]
    
    # Filter the ticks so we only plot ones that fall within your actual data range
    valid_ticks = [t for t in custom_ticks if min(y_start, y_end) <= t <= max(y_start, y_end)]
    
    ax.set_yticks(valid_ticks)
    
    # 4. Use a string formatter to display decimal points nicely (or drop them for integers)
    # This prevents Matplotlib from breaking formatting on fractional levels like 0.1 or 0.5
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:g}"))

    # 5. Complete labels and clean grid lines
    ax.set_title(f"Max Cross-Correlation Lag Profile (Base: {base_plev/100.0} hPa)")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Pressure (hPa)")
    
    # turn on both major and minor grids for log scale visibility
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    plt.show()

    # OG Code
    # data = xr.load_dataset(path, engine="netcdf4")
    # data = data.sel(time=slice(start, end)).mean(dim="lat")
    # print(data)
    # start_idx = data.indexes['plev'].get_loc(base_plev)
    # plevs = data.isel(plev=slice(start_idx + 1, None)).plev.values
    # lons = data.lon.values
    # max_xcor = []
    # # Iterate per stripe
    # for lon in lons:
    #     # Calculate base
    #     base = data.sel(plev=base_plev, lon=lon)
    #     base_smooth = base["CO2"].rolling(time=12, center=True).mean().dropna(dim="time")
    #     # Compare pressure levels
    #     for plev in plevs:
    #         plev_data = data.sel(plev=plev, lon=lon)
    #         plev_smooth = plev_data["CO2"].rolling(time=12, center=True).mean().dropna(dim="time")
    #         base_align, plev_align = xr.align(base_smooth, plev_smooth)
    #         base_values = base_align.diff(dim="time").values
    #         plev_values = plev_align.diff(dim="time").values

    #         lags = signal.correlation_lags(len(plev_values), len(base_values))
    #         ccf = ccf_values(plev_values, base_values)
    #         max_idx = np.argmax(ccf)
    #         max_xcor.append(lags[max_idx])
    # print(max_xcor)



def ccf_table_optimized(path, base_plev, start, end):
    ## AI optimized version
    data = xr.load_dataset(path, engine="netcdf4")
    data = data.sel(time=slice(start, end)).mean(dim="lon")
    print(data)

    start_idx = data.indexes["plev"].get_loc(base_plev)
    plevs = data.isel(plev=slice(start_idx + 1, None)).plev.values
    lats = data.lat.values

    # 1. Pre-compute rolling means globally, but DO NOT drop NaNs yet.
    # This removes 95% of the slow Xarray overhead while preserving edge variations.
    global_smooth = data["CO2"].rolling(time=12, center=True).mean()

    # Split into base and targets globally to speed up .sel() lookups
    global_base = global_smooth.sel(plev=base_plev)
    global_targets = global_smooth.sel(plev=plevs)

    max_xcor = []

    # 2. Iterate per stripe using fast localized slices
    for lat in lats:
        # Extract base and drop NaNs dynamically for this specific longitude
        base_smooth = global_base.sel(lat=lat).dropna(dim="time")

        for plev_idx, plev in enumerate(plevs):
            # Extract target and drop NaNs dynamically for this specific level/longitude
            plev_smooth = (
                global_targets.isel(plev=plev_idx).sel(lat=lat).dropna(dim="time")
            )

            # Your original alignment and math remain 100% untouched
            base_align, plev_align = xr.align(base_smooth, plev_smooth)
            base_values = base_align.diff(dim="time").values
            plev_values = plev_align.diff(dim="time").values

            lags = signal.correlation_lags(len(plev_values), len(base_values))
            ccf = ccf_values(plev_values, base_values)
            max_idx = np.argmax(ccf)
            max_xcor.append(lags[max_idx])

    print(max_xcor)
    return max_xcor


# def ccf_table_to_xarray(path, base_plev, start, end):
#     data = xr.load_dataset(path, engine="netcdf4")
#     data = data.sel(time=slice(start, end)).mean(dim="lat")
#     print(data)

#     start_idx = data.indexes["plev"].get_loc(base_plev)
#     plevs = data.isel(plev=slice(start_idx + 1, None)).plev.values
#     lons = data.lon.values

#     # Pre-compute rolling means globally without dropping NaNs yet
#     global_smooth = data["CO2"].rolling(time=12, center=True).mean()
#     global_base = global_smooth.sel(plev=base_plev)
#     global_targets = global_smooth.sel(plev=plevs)

#     # Initialize a 2D NumPy grid to store lag values: shape (plev, lon)
#     grid_shape = (len(plevs), len(lons))
#     max_xcor_grid = np.empty(grid_shape)

#     # Iterate over longitudes and pressure levels
#     for lon_idx, lon in enumerate(lons):
#         base_smooth = global_base.sel(lon=lon).dropna(dim="time")

#         for plev_idx, plev in enumerate(plevs):
#             plev_smooth = (
#                 global_targets.isel(plev=plev_idx).sel(lon=lon).dropna(dim="time")
#             )

#             base_align, plev_align = xr.align(base_smooth, plev_smooth)
#             base_values = base_align.diff(dim="time").values
#             plev_values = plev_align.diff(dim="time").values

#             lags = signal.correlation_lags(len(plev_values), len(base_values))
#             ccf = ccf_values(plev_values, base_values)
#             max_idx = np.argmax(ccf)
            
#             # Map the exact lag directly to its grid coordinate
#             max_xcor_grid[plev_idx, lon_idx] = lags[max_idx]

#     # Convert the raw grid into a clean, graph-ready Xarray DataArray
#     da_out = xr.DataArray(
#         data=max_xcor_grid,
#         dims=["plev", "lon"],
#         coords={"plev": plevs, "lon": lons},
#         name="max_lag",
#         attrs={"description": f"Cross-correlation lag relative to base_plev {base_plev}", "units": "months"}
#     )
    
#     return da_out


def ccf_table_to_xarray(path, base_plev, start, end):
    data = xr.load_dataset(path, engine="netcdf4")
    # 1. Flatten the longitude dimension instead of latitude
    data = data.sel(time=slice(start, end)).mean(dim="lon")
    print(data)

    start_idx = data.indexes["plev"].get_loc(base_plev)
    plevs = data.isel(plev=slice(start_idx + 1, None)).plev.values
    # 2. Extract latitude coordinates instead of longitude
    lats = data.lat.values

    # Pre-compute rolling means globally without dropping NaNs yet
    global_smooth = data["CO2"].rolling(time=12, center=True).mean()
    global_base = global_smooth.sel(plev=base_plev)
    global_targets = global_smooth.sel(plev=plevs)

    # 3. Update the 2D grid shape to use latitude: shape (plev, lat)
    grid_shape = (len(plevs), len(lats))
    max_xcor_grid = np.empty(grid_shape)

    # 4. Iterate over latitudes and pressure levels
    for lat_idx, lat in enumerate(lats):
        base_smooth = global_base.sel(lat=lat).dropna(dim="time")

        for plev_idx, plev in enumerate(plevs):
            plev_smooth = (
                global_targets.isel(plev=plev_idx).sel(lat=lat).dropna(dim="time")
            )

            if len(base_smooth.time) == 0 or len(plev_smooth.time) == 0:
                max_xcor_grid[plev_idx, lat_idx] = np.nan
                continue

            base_align, plev_align = xr.align(base_smooth, plev_smooth)
            base_values = base_align.diff(dim="time").values
            plev_values = plev_align.diff(dim="time").values

            lags = signal.correlation_lags(len(plev_values), len(base_values))
            ccf = ccf_values(plev_values, base_values)

            non_negative_mask = lags >= 0
            
            # Filter both arrays to only evaluate forward-moving time
            valid_lags = lags[non_negative_mask]
            valid_ccf = ccf[non_negative_mask]
            
            # Find the maximum index within your filtered subset
            max_idx = np.argmax(valid_ccf)
            
            # Map the exact lag directly to its grid coordinate
            max_xcor_grid[plev_idx, lat_idx] = valid_lags[max_idx]

    # 5. Convert into a clean Xarray DataArray labeled with latitude coordinates
    da_out = xr.DataArray(
        data=max_xcor_grid,
        dims=["plev", "lat"],
        coords={"plev": plevs, "lat": lats},
        name="max_lag",
        attrs={"description": f"Cross-correlation lag relative to base_plev {base_plev}", "units": "months"}
    )
    
    return da_out




def plot_ccf(path, plev, zec=False):
    data = xr.load_dataset(path, engine="netcdf4")

    lags_plev, ccf_plev, data_len, base = do_calc(data, plev)
    
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.set_title(f'Cross Correlation of CO2 at {plev/100} hPa vs Baseline {base/100} hPa', weight='bold', fontsize = 15)
    create_ccf_plot(lags_plev, ccf_plev, data_len, ax, [-0.75, 0.75], "green")
    
    # some additional analysis 
    max_idx = np.argmax(ccf_plev)
    exact_lag = lags_plev[max_idx]
    max_corr = ccf_plev[max_idx]
    print(f"The maximum correlation of {max_corr:.3f} occurs at a lag of {exact_lag} timesteps.")
    
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
    # years = np.arange(1920, 1946, 5, dtype=int)
    # print(years)
    # months = [9] # np.arange(1, 13, dtype=int)
    # print(months)
    # plot_ccf(path, args.plev, args.zec)
    # plevs = [8.50e+04, 7.75e+04, 7.00e+04, 6.00e+04, 5.00e+04,
    #    4.00e+04, 3.00e+04, 2.50e+04, 2.00e+04, 1.50e+04, 1.00e+04, 7.00e+03,
    #    5.00e+03, 3.00e+03, 2.00e+03, 1.00e+03, 7.00e+02, 5.00e+02, 3.00e+02,
    #    2.00e+02, 1.00e+02, 5.00e+01, 2.00e+01, 1.00e+01]
    # just_math_ccf(path, plevs)
    ccf_table(path, 9.25e+04, "1900-01", "1920-12")
