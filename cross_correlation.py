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
    for plev in plevs:
        lags_plev, ccf_plev, data_len, base = do_calc(data, plev)
        max_idx = np.argmax(ccf_plev)
        exact_lag = lags_plev[max_idx]
        max_corr = ccf_plev[max_idx]
        print(f"{plev/100} hPa: Max corr {max_corr:.3f} at {exact_lag} Months")


def do_calc(data, plev):
    # compare to baseline
    base = 9.25e+04
    start_time = '1900-01-31T23:52:00.000000000'
    end_time = '1930-12-31T23:52:00.000000000'
    data = data.sel(time=slice(start_time, end_time))

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
    parser.add_argument("plev", type=int, help="pressure level")
    parser.add_argument("--zec", action="store_true", help="use zec data")

    args = parser.parse_args()
    log_args(args)

    path = '/media/lisa/PortableSSD/Data/flat10/HW310_esm-flat10_RESCUE/echam6/HW310_esm-flat10_RESCUE_echam6_tracerpl_1850-1949.nc'
    if (args.zec):
        print("NOT SET UP FOR ZEC DATA YET")
        # path = '/media/lisa/PortableSSD/Data/flat10/HW312_esm-flat10-zec_RESCUE/echam6/'
    print(path)
    years = np.arange(1920, 1946, 5, dtype=int)
    # print(years)
    # months = [9] # np.arange(1, 13, dtype=int)
    # print(months)
    # plot_ccf(path, args.plev, args.zec)
    plevs = [8.50e+04, 7.75e+04, 7.00e+04, 6.00e+04, 5.00e+04,
       4.00e+04, 3.00e+04, 2.50e+04, 2.00e+04, 1.50e+04, 1.00e+04, 7.00e+03,
       5.00e+03, 3.00e+03, 2.00e+03, 1.00e+03, 7.00e+02, 5.00e+02, 3.00e+02,
       2.00e+02, 1.00e+02, 5.00e+01, 2.00e+01, 1.00e+01]
    just_math_ccf(path, plevs)