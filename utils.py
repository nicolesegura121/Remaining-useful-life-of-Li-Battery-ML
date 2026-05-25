#Important functions for the project on battery managment
def time_format(meta):
    import pandas as pd
    #Cleaning the format from symbols
    meta["start_time"]=meta["start_time"].replace((r'[\[\]]'), '',regex=True)
    #Split in columns
    meta[["year","month","day","hour","minute","second"]]=meta["start_time"].str.split(n=5,expand=True)
    #Transforming exponential notations to decimals

    meta[["year","month","day","hour","minute","second"]]=meta[["year","month","day","hour","minute","second"]].astype(float)
    #some seconds have microseconds 
    meta["microsecond"] = (meta["second"]- meta["second"].astype(int))*1000000
    meta["new_time"] = pd.to_datetime({
    "year":meta["year"].astype(int),
    "month":meta["month"].astype(int),
    "day":meta["day"].astype(int),
    "hour":meta["hour"].astype(int),
    "minute":meta["minute"].astype(int),
    "second":meta["second"].astype(int),
    "microsecond" : meta["microsecond"].astype(int)
    })

    meta=meta.drop(["start_time","year","month","day","hour","minute","second","microsecond"],axis=1)
    return meta

def preprocessing_meta(meta): 
    import pandas as pd
    #Converting Capacity and resistances to integer values
    meta["Capacity"]=pd.to_numeric(meta["Capacity"],errors="coerce")
    meta["Re"]=pd.to_numeric(meta["Re"],errors="coerce")
    meta["Rct"]=pd.to_numeric(meta["Rct"],errors="coerce")
    return meta

def battery_cycles_df(meta,battery_names, types): 
    import pandas as pd
    """ 
    The goal is to extract all relevant data in a single dataframe, whether it is charge, discharge or impedance.
    meta = (df) original dataframe that contains the file names, and battery ids etc
    battery_name = as given in meta 
    type = (list) charge, discharge or impedance
    return: single dataframe concatenated
    """
    if isinstance(battery_names, str):
        battery_names = [battery_names]

    if isinstance(types, str):
        types = [types]
        
    condition = ((meta["battery_id"].isin(battery_names)) & (meta["type"].isin(types)))
    selection_df = meta.loc[condition]

    cycles = [] #empty list to store the data

    for _, row in selection_df.iterrows():
        filepath = row["filename"] # I have the file name for each row of charge or discharge     
        data = pd.read_csv(f"data/{filepath}") #and I read each file

        data["type"] = row["type"]
        data["test_id"] = row["test_id"]
        data["battery_id"] = row["battery_id"]
        data["start_time"] = row["new_time"]

        if "impedance" in types:
            data["Re"] = row["Re"]
            data["Rct"] = row["Rct"]   

            cycles.append(data)
        
        elif "discharge" in types:
            data["Capacity"] = row["Capacity"]
            cycles.append(data)
        else: 
            cycles.append(data)
            
    combined_df = pd.concat(cycles, ignore_index=True).reset_index(drop=True)
    return combined_df

def continuous_time(df):
    # Convert datetime to Unix seconds
    unix_start = df["start_time"].astype("int64") / 1e9

    # Absolute continuous experiment time
    df["real_time"] = unix_start + df["Time"]

    # Normalize EACH battery independently
    df["norm_time"] = (df.groupby("battery_id")["real_time"].transform(lambda x: x - x.min())).reset_index(drop=True)

    return df

def current_capacity(df):
    import numpy as np
    # Ensure proper ordering
    df = df.sort_values(["battery_id", "test_id", "norm_time"])

    #Time differential per battery using the normalized time
    df["dt"]= df.groupby(["battery_id", "test_id"])["norm_time"].diff().fillna(0)
    
    #Incremental change in capacity
    df["dQ"] = df["Current_measured"]*df["dt"]
    
    #cumulative capacity

    df["capacity_Ah"] = df.groupby(["battery_id", "test_id"])["dQ"].cumsum()/3600
    df=df.drop(["dt","dQ"],axis=1).reset_index(drop=True)
    return df

def visualization_2by2(df,x,y):
    import matplotlib.pyplot as plt

    fig,ax = plt.subplots(2,2,figsize=(12,8))

    ax = ax.flatten()

    batteries = df["battery_id"].unique()

    cmap = plt.cm.viridis

    for index,battery in enumerate(batteries):
    
        battery_df = df[df["battery_id"]==battery]
        cycles = battery_df["test_id"].unique()[::40]

        for j,cycle in enumerate(cycles):
            color = cmap(j / (len(cycles)-1))
            cycle_df = battery_df[battery_df["test_id"] == cycle]

            ax[index].plot(cycle_df[x],cycle_df[y],label=f"{cycle}",color=color)

            ax[index].set_title(battery)

            ax[index].set_xlabel(f"{x}")
            ax[index].set_ylabel(f"{y}")
            ax[index].legend(title="Cycle")
    return ax[index]


def area_under_curve(df,x, y):

    import numpy as np
    import pandas as pd
    """
    This function will return capacity based on the current and time data provided. 
    Q=(integral) = Idt  (area under the curve if time is x and current y)
    Assuming that current is given in amps and time in seconds, the function returns the capcity in Ah.
    """
    results = []

    grouped = df.groupby(["battery_id", "test_id"])

    for (battery, cycle), cycle_df in grouped:
        cycle_df = cycle_df.sort_values("Time")
        
        X = cycle_df[x].values

        Y = cycle_df[y].values

        A = np.trapezoid(Y,X)
        
        results.append({
            "battery_id": battery,
            "test_id": cycle,
            "Area_Under_curve": A,
        })

    return pd.DataFrame(results)

def timeTmax(df):

    from pybaselines import Baseline
    from scipy.signal import find_peaks
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit
    from scipy.ndimage import gaussian_filter1d

    results_Tmax = []

    grouped = df.groupby(["battery_id", "test_id"])
    #setting default values to avoid mismatch with initial df
    for (battery, cycle), cycle_df in grouped:
        t_peak = np.nan
        T_peak = np.nan
        try: 
        

            cycle_df = cycle_df.sort_values("Time")
            temp = gaussian_filter1d(cycle_df["Temperature_measured"].values,sigma=10)
            time = cycle_df["Time"].values

            # Skip tiny cycles
            if len(time) >=  20:

                def exp_decay(x, A, k, c):
                    return A * np.exp(-k * x) + c
        
                c0= cycle_df["Temperature_measured"].iloc[-1]
                A0 = cycle_df["Temperature_measured"].iloc[0]-c0
                duration = cycle_df["Time"].max()
                k0 = 1/ duration

                p0 = (A0, k0, c0)

                mask = ((time < 500) | (time > 7000))
                params, _ = curve_fit(
                    exp_decay,
                    time[mask],
                    temp[mask],
                    p0=p0,
                    maxfev=10000)

                baseline_exp = exp_decay(time, *params)

                y = temp - baseline_exp

                peaks, properties = find_peaks(y,prominence=1)

                if (len(peaks) > 0 or len(properties["prominences"]) > 0):
                    main_peak = peaks[np.argmax(properties["prominences"])]

                    t_peak = time[main_peak]

                    T_peak = y[main_peak]
        except:
            pass


        results_Tmax.append({
            "battery_id": battery,
            "test_id": cycle,
            "time_Tmax": t_peak,
            "T_peak": T_peak})
            
    return pd.DataFrame(results_Tmax)

def time_Tmax3(df,t_min=400):
    import pandas as pd
    results = []
    grouped = df.groupby(["battery_id", "test_id"])
    
    for (battery, cycle), cycle_df in grouped:

        cycle_df = cycle_df.sort_values("Time")

        cycle_df = cycle_df[cycle_df["Time"]>=t_min]

        if cycle_df.empty: 
            continue
        idx_Tmax = cycle_df["Temperature_measured"].idxmax()
        T_max = cycle_df.loc[idx_Tmax,"Temperature_measured"]
        time_Tmax = cycle_df.loc[idx_Tmax,"Time"]
        
        results.append({
            "battery_id": battery,
            "test_id": cycle,
            "time_Tmax": time_Tmax,
            "Tmax": T_max
        })
    return pd.DataFrame(results)

def exp_decay(x, A, k, c):
    import numpy as np
    return A * np.exp(-k * x) + c


def time_to_voltage(df, voltage_target = 4.2, tolerance = 0.05):
    import pandas as pd
    import numpy as np
    results = []
    grouped = df.groupby(["battery_id", "test_id"])
    for (battery, cycle), cycle_df in grouped:

        cycle_df = cycle_df.sort_values("Time")
        voltage = cycle_df["Voltage_measured"]
        mask = (np.abs(voltage-voltage_target))<=tolerance

        #required to sustain the value
        window=3
        rolling_hits = (mask.rolling(window).sum())
        idx = rolling_hits[rolling_hits==window].index

        
        if len(idx) ==0:
            t_reach= np.nan
        else: 
            first_idx = idx[0]
            t_reach=cycle_df.loc[first_idx,"Time"]
        results.append({
            "battery_id": battery,
            "test_id": cycle,
            "time_to_4p2V": t_reach
        })
    return pd.DataFrame(results)

def charging_decay(df):
    import pandas as pd
    from scipy.optimize import curve_fit
    import numpy as np
    
    results = []
    
    grouped = df.groupby(["battery_id", "test_id"])
    
    for (battery, cycle), cycle_df in grouped:
        #To avoid loosing rows: 
        exp_coeff = np.nan

        try: 
            cycle_df = cycle_df.sort_values("Time")
            voltage = cycle_df["Voltage_measured"]
            mask = (np.abs(voltage-4.2))<= 0.02

            #required to sustain the value
            window=5
            rolling_hits = (mask.rolling(window).sum())
            idx = rolling_hits[rolling_hits==window].index

            if len(idx) > 0:
                first_idx = idx[0]
                t_to42V=cycle_df.loc[first_idx,"Time"]
            
            #Define the CV phase

            CV_phase = cycle_df.loc[((cycle_df["Time"]>=t_to42V)&(cycle_df["Voltage_measured"]>4.18))] 

            if len(CV_phase)>=10:

                CV_phase["t_rel"] = (CV_phase["Time"] - CV_phase["Time"].iloc[0])
                
                #remove invalid values

                CV_phase = CV_phase.replace([np.inf, -np.inf],np.nan)
                
                CV_phase = CV_phase.dropna(subset=["Current_measured","t_rel"])

                if len(CV_phase) >= 10:
                    c0 = CV_phase["Current_measured"].iloc[-1]
                    A0 = (CV_phase["Current_measured"].iloc[0]-c0)
                    duration = CV_phase["t_rel"].max()
                    if duration >0: 

                        k0 = 1/duration
                    
                        popt, pcov = curve_fit(exp_decay, CV_phase["t_rel"], CV_phase["Current_measured"], p0=(A0, k0, c0),maxfev=10000)

                        exp_coeff = popt[1]
        except: 
            pass
        results.append({
            "battery_id": battery,
            "test_id": cycle,
            "exp_coefficient": exp_coeff,
        })
    return pd.DataFrame(results)

def charging_slope(df):

    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import linregress
    import pandas as pd
    import numpy as np

    results = []

    grouped = df.groupby(["battery_id", "test_id"])

    for (battery, cycle), cycle_df in grouped:
        #Default
        slope = np.nan
        intercept = np.nan
        try: 

            cycle_df = cycle_df.sort_values("Time")

            #Smooth out the curve
            cycle_df["V_smooth"] = gaussian_filter1d(cycle_df["Voltage_measured"],sigma=10)

            #Calculate derivates to estimate change of curve
            cycle_df["dVdt"] = cycle_df["V_smooth"].diff()/cycle_df["Time"].diff()
            cycle_df["d2Vd2t"] = cycle_df["dVdt"].diff()/cycle_df["Time"].diff()

            #Mask the curve
            d2Vd2t_mask =  ((cycle_df["d2Vd2t"]>=-1e-6) & (cycle_df["d2Vd2t"]<= 2e-6))
            V_mask = (d2Vd2t_mask & (cycle_df["Voltage_measured"]< 4.18))
            V_mask2 = (V_mask & (cycle_df["Voltage_measured"]> 3.8))
            V_mask3 = (V_mask2 & (cycle_df["Current_measured"]> 1))
                
                #Extract the increasing slope
            slopeV = cycle_df[V_mask3].copy()

            if len(slopeV) >=2:
                x = slopeV["capacity_Ah"]

                y = slopeV["Voltage_measured"]

                #Remove nans
                valid = (np.isfinite(x) & np.isfinite(y))

                x = x[valid]
                y = y[valid]

            if len(x)>=2: 
                res = linregress(x, y)
                slope= res.slope
                intercept = res.intercept
        except: 
            pass

      
        results.append({
            "battery_id": battery,
            "test_id": cycle,
            "slope": slope,
            "intercept": intercept
            })
    return pd.DataFrame(results)

def ransac_clean_outliers(df,feature):
    from sklearn.linear_model import (RANSACRegressor, LinearRegression)
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    import numpy as np
    import pandas as pd
    df_clean = df.copy()

    df_clean[f"{feature}_outlier"] = False

    batteries = df_clean["battery_id"].unique()
    model = make_pipeline(PolynomialFeatures(4),LinearRegression())
    threshold= {
        "time_to_4p2V" : 510,
        "time_Tmax" : 350,
        "T_peak" : 1,
        "exp_coefficient":1e-4,
        "Area_Under_curve":10000,
        "slope":0.5,
        "intercept": 0.5,
        "SoH": 30,
    }
    for battery in batteries: 
        mask_battery = (df_clean["battery_id"]==battery)
        battery_df = df_clean.loc[mask_battery].copy()
        x= battery_df["test_id"].values.reshape(-1,1)
        y= battery_df[feature].values

        ransac = RANSACRegressor(estimator=model, random_state=42, min_samples= 4, residual_threshold = threshold[feature]).fit(x,y)
        y_pred = ransac.predict(x)
        outlier_mask = (~ransac.inlier_mask_)
        outlier_idx = battery_df.index[outlier_mask]
        

        #marking outliers
        df_clean.loc[outlier_idx, f"{feature}_outlier"]=True
        #replace outliers with corrected values
        corrected_values = (y_pred[outlier_mask])
        #interpolate those values
        df_clean.loc[outlier_idx,feature] = corrected_values
    return df_clean
