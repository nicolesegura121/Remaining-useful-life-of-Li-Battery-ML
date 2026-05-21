#Important functions for the project on battery managment
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
            
    combined_df = pd.concat(cycles, ignore_index=True)
    return combined_df

def continuous_time(df):
    # Convert datetime to Unix seconds
    unix_start = df["start_time"].astype("int64") / 1e9

    # Absolute continuous experiment time
    df["real_time"] = unix_start + df["Time"]

    # Normalize EACH battery independently
    df["norm_time"] = (df.groupby("battery_id")["real_time"].transform(lambda x: x - x.min()))

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
    df=df.drop(["dt","dQ"],axis=1)
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


def total_capacity(time, current):
    import numpy as np
    """
    This function will return capacity based on the current and time data provided. 
    Q=(integral) = Idt  (area under the curve if time is x and current y)
    Assuming that current is given in amps and time in seconds, the function returns the capcity in Ah.
    """
    Q = np.trapezoid(time, current)
    return Q

