import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal


def enso_index(compare=None, name=None):
    path = '/media/lisa/PortableSSD/Data/flat10/HW310_esm-flat10_RESCUE/echam6/HW310_esm-flat10_RESCUE_echam6_tsw_1850-1949.nc'
    data = xr.open_dataset(path, engine="netcdf4")
    nino_region = data.sel(lat=slice(5, -5), lon=slice(190, 240))
    nino_mean = nino_region.mean(dim=['lat', 'lon'])
    tsw_cleaned = nino_mean['tsw'].where(nino_mean['tsw'] > 0)
    # Index is determined by a rolling 3-month anomaly
    three_month_rolling = tsw_cleaned.rolling(time=3, center=True).mean()
    # Compare to average over the entire time period or first decades?
    # .sel(time=slice('1850-01-01', '1880-12-31'))
    total_average = tsw_cleaned.mean(dim='time')
    anomalies = three_month_rolling - total_average
    
    anomalies.plot.line(label='Nino 3.4 Index')
    print(anomalies.time)

    time_coords = anomalies.time.values
    anomaly_values = anomalies.values

    # 4. Add the El Niño shading (above 0.5)
    plt.fill_between(time_coords, anomaly_values, 0.5,
                    where=(anomaly_values >= 0.5),
                    color='red', alpha=0.3, interpolate=True, label='El Niño Active')

    # 5. Optional: Add La Niña shading (below -0.5)
    plt.fill_between(time_coords, anomaly_values, -0.5,
                    where=(anomaly_values <= -0.5),
                    color='blue', alpha=0.3, interpolate=True, label='La Niña Active')

    # # 6. Reference horizontal threshold lines
    # plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    # plt.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    # plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    if compare is not None:
        print("Adding comparison data.")
        nino_xcorr(anomalies, compare, name, units='months')

        # datetime_fixed = pd.to_datetime([f"{int(y)}-06-30" for y in compare.year.values])
        # datetime_fixed = pd.to_datetime(compare.time.values) + pd.offsets.MonthEnd(0)
        # compare.plot.line(x='year', label=name if name else 'Comparison', color='green')
        plt.plot(compare, compare.values, label=name if name else 'Comparison', color='green')
        plt.title(f'Nino 3.4 Index vs {name}' if name else 'Nino 3.4 Index Comparison')
        plt.ylabel('Temperature Anomaly (K) / CO2 Change (ppm 10e12)')
    else:
        plt.title('Nino 3.4 Index (3-month rolling anomaly)')
        plt.ylabel('Temperature Anomaly (K)')

    plt.xlabel('Year')
    plt.legend()
    plt.grid()
    plt.show()

    return anomalies


# This is mostly AI-generated
def nino_xcorr(anomalies, compare, name, units=None):
    if units == 'months':
        s_nino = pd.Series(anomalies.values, index=pd.to_datetime(anomalies.time.values).to_period('M'))
        s_compare = pd.Series(compare.values, index=pd.to_datetime(compare.time.values).to_period('M'))
    else:
        # Downsample your monthly anomalies to an annual average
        nino_annual = anomalies.resample(time='1YE').mean()

        # Convert both to simple pandas Series to make alignment easy
        s_nino = pd.Series(nino_annual.values, index=nino_annual.time.dt.year.values)
        s_compare = pd.Series(compare.values, index=compare.year.values)

    # Drop any NaNs that might break the math
    s_nino = s_nino.dropna()
    s_compare = s_compare.dropna()

    # Find the overlapping years between both datasets
    common_years = s_nino.index.intersection(s_compare.index)
    y1 = s_nino.loc[common_years].values
    y2 = s_compare.loc[common_years].values

    # --- STEP 2: Normalize the data (Crucial for correlation scaling) ---
    y1_norm = (y1 - np.mean(y1)) / (np.std(y1) * len(y1))
    y2_norm = (y2 - np.mean(y2)) / np.std(y2)

    # --- STEP 3: Compute Cross-Correlation ---
    # This calculates correlation coefficients for all possible shifts
    corr = signal.correlate(y1_norm, y2_norm, mode='full')
    lags = signal.correlation_lags(len(y1), len(y2), mode='full')

    # Find the maximum correlation value and its corresponding lag
    max_corr_idx = np.argmax(np.abs(corr))
    best_lag = lags[max_corr_idx]
    best_corr = corr[max_corr_idx]

    print(f"Strongest correlation: {best_corr:.2f} at a lag of {best_lag} units.")

    # --- STEP 4: Plot the Cross-Correlation Function (CCF) ---
    plt.figure(figsize=(10, 4))
    plt.stem(lags, corr)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero Lag')
    plt.xlabel('Lag (Months)')
    plt.ylabel('Correlation Coefficient')
    plt.title(f'Cross-Correlation: Niño 3.4 vs {name if name else "Comparison"}')
    plt.xlim(-120, 120) # Zoom into a reasonable window of shifts
    plt.minorticks_on()
    plt.grid(visible=True, which='major', color='#d3d3d3', linestyle='-', linewidth=0.8)
    plt.grid(visible=True, which='minor', color='#e0e0e0', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    anomalies = enso_index()
    
    # el_nino_records = {}

    # # Group by year to ensure we process one calendar block at a time
    # for year, group in anomalies.groupby('time.year'):
    #     peak_val = float(group.max())
        
    #     # Check if the peak of this specific year crossed the threshold
    #     if peak_val > 1.5:
    #         el_nino_records[year] = peak_val

    # print(f"{'Super Year':<8} | {'Peak Anomaly (°C)':<18}")
    # print("-" * 30)

    # for year, val in zip(el_nino_records.keys(), el_nino_records.values()):
    #     print(f"{year:<8} | +{val:.2f}°C")
