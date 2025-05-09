import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import scipy
import numpy as np
import networkx as nx
from scipy import stats


# Run the code from terminal
# streamlit run C:\Users\Simon\PycharmProjects\pythonProject\.sem3\fordypning\Master\ALL\Streamlit_PS.py

# Define a function to get available files
def get_available_files(directory, filetype):
    return [f for f in os.listdir(directory) if f.endswith(filetype)]


# Collect desired file from folder
def get_file(file, directory):
    file_path = os.path.join(directory, file)
    df = None
    if file.endswith('.csv'):
        df = pd.read_csv(file_path)
    if file.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    return df


# Initial Variables
previous_error = 0.0
integral = 0.0

# Fuzzy logic PID-Values
# Kp = 0.1845
# Ki = 0.0052
# Kd = 0.0054

# Personalized numbers, more dynamic for multiple stations
Kp = 0.184
Ki = 0.0055
Kd = 0.005


# Folder Directories
# RAW_DATA_DIR = 'C:/Users/Simon/PycharmProjects/pythonProject/.sem3/fordypning/Master/ALL/PA2_split'
RAW_DATA_DIR = None
DATA_DIR = './'
COST_DIR = './poly_win'
COST_DIR_FUTURE = './poly_win_2100'
RAIN_DIR = './Rain'

parquet_files = get_available_files(DATA_DIR, '.parquet')
# raw_files = get_available_files(RAW_DATA_DIR,'.csv')
raw_files = None
rain_files = get_available_files(RAIN_DIR, '.csv')

# Individual files used
total_nan = pd.read_csv('./NAN_folder/nan_summary.csv')

# Streamlit UI
st.title('Station Viewer')
st.sidebar.header('Station mode')

mode_type = st.sidebar.radio('', ['Raw data', 'Climate change', 'Station flow', 'Station control', 'Maintenance', 'Station consumption', 'Userdefined analysis'])


def main():
    if mode_type == 'Raw data':
        rawdata(raw_files, parquet_files, rain_files)
    elif mode_type == 'Climate change':
        climate(rain_files)
    elif mode_type == 'Station flow':
        flow(rain_files)
    elif mode_type == 'Station control':
        control(parquet_files)
    elif mode_type == 'Maintenance':
        maintenance()
    elif mode_type == 'Station consumption':
        consumption()
    elif mode_type == 'Userdefined analysis':
        user_data(rain_files)


# Main code for the raw data UI
def rawdata(csv_files, parq_files, rain_file):
    st.header('Raw data')
    st.write('Raw data was too large and will not be included')
    # selected_raw_file = st.selectbox('Choose available file', csv_files)

    # if selected_raw_file:
    #     df_raw = get_file(selected_raw_file, RAW_DATA_DIR)
    #     st.write('### Preview of', selected_raw_file)
    #     col1_1, col1_2 = st.columns(2)
    #     col1_1.write('Average station tank level')
    #     col1_1.dataframe(df_raw[['timestamp', 'avg']].rename(columns={'avg': 'level'}), hide_index=True, width=500)
    #     col1_2.write('Nan-values')
    #     col1_2.dataframe(df_raw[['timestamp', 'avg']].isna().sum().to_frame(name='value'))

    st.header('Total nan-values')
    df_summary = total_nan
    data_one_year = 60 * 24 * 365
    data_leap_year = 60 * 24 * 366

    df_summary['2022_%'] = (df_summary['2022']/data_one_year) * 100
    df_summary['2023_%'] = (df_summary['2023']/data_one_year) * 100
    df_summary['2024_%'] = (df_summary['2024']/data_leap_year) * 100
    df_summary['2025_%'] = (df_summary['2025']/91567) * 100

    df_summary = df_summary[['filename', '2022', '2022_%','2023', '2023_%','2024', '2024_%','2025', '2025_%']]
    df_summary = df_summary.style.map(lambda x: 'background-color: green', subset=(slice(0, 8, 1), ['filename', '2023', '2023_%', '2024', '2024_%'])).format(precision=2)
    st.dataframe(df_summary, hide_index=True, width=650)

    st.header('Rainfall data')

    rain_2022 = get_file(rain_file[0], RAIN_DIR)
    rain_2023 = get_file(rain_file[1], RAIN_DIR)
    rain_2024 = get_file(rain_file[2], RAIN_DIR)

    config = {'timestamp': st.column_config.DateColumn('timestamp', format='YYYY-MM-DD')}
    col2_1, col2_2, col2_3 = st.columns(3)
    col2_1.dataframe(time_convert(rain_2022), column_config=config, hide_index=True)
    col2_2.dataframe(time_convert(rain_2023), column_config=config, hide_index=True)
    col2_3.dataframe(time_convert(rain_2024), column_config=config, hide_index=True)

    st.header('Processed data')
    selected_file = st.selectbox('Choose available file', parq_files)
    if selected_file:
        df_processed = get_file(selected_file, DATA_DIR)
        st.write('### Preview of', selected_file)
        st.dataframe(df_processed.rename(columns={'avg': 'level'}), hide_index=True, width=650)


# Main code for the climate analysis
def climate(rain_file):
    df = get_file(rain_file[3], RAIN_DIR)

    df_historical = historical_rain(df)

    st.header('Climate change in Møre og Romsdal')
    st.subheader('Historical rain for Møre og Romsdal:')

    col1, col2 = st.columns(2)
    historical_min = df_historical[df_historical['rain'] == df_historical['rain'].min()][['year', 'rain']]
    historical_max = df_historical[df_historical['rain'] == df_historical['rain'].max()][['year', 'rain']]
    historical_mean = df_historical['rain'].mean()

    col1.write(f"Lowest yearly rainfall was recorded in {historical_min['year'].iloc[0]} which measured at {historical_min['rain'].iloc[0]} mm")
    col1.write(f"Highest yearly rainfall was recorded in {historical_max['year'].iloc[0]} which measured at {historical_max['rain'].iloc[0]} mm")
    col1.write(f'Average yearly rainfall for the period 1958-2024 is {historical_mean:.1f} mm')
    col2.dataframe(df_historical, hide_index=True, height=200, width=200)

    values_1958 = regression(df_historical)
    values_2000 = regression(df_historical[['year', 'rain']].iloc[42:])

    # slope = [0], intercept = [1]
    scipy_line_1958 = scipy_line_historical(slope=values_1958[0], intercept=values_1958[1], df=(df_historical['year']))
    scipy_line_2000 = scipy_line_historical(slope=values_2000[0], intercept=values_2000[1], df=(df_historical['year'].iloc[42:]))

    scipy_line_1958_future = scipy_line_future(slope=values_1958[0], intercept=values_1958[1], year=2100)
    scipy_line_2000_future = scipy_line_future(slope=values_2000[0], intercept=values_2000[1], year=2100)

    baseline = scipy_line_1958.iloc[-1]
    future_period_1 = np.arange(2025, 2071)
    future_period_2 = np.arange(2071, 2100)

    rcp45_proj_1, rcp45_proj_2, rcp85_proj_1, rcp85_proj_2 = rcp(future_period_1, future_period_2, baseline)

    col1_1, col1_2, col1_3, col1_4 = st.columns(4)
    plot_45 = col1_1.checkbox('RCP 4.5')
    plot_85 = col1_2.checkbox('RCP 8.5')
    plot_tl_1958 = col1_3.checkbox('Trend (1958-2024)')
    plot_tl_2000 = col1_4.checkbox('Trend (2000-2024)')

    fig, ax = plt.subplots(figsize=(14, 8))
    rcp45_color = 'orange'
    rcp85_color = 'red'

    ax.plot(df_historical['year'], df_historical['rain'], label='Historical data', color='black', linewidth=2)
    if plot_tl_1958:
        ax.plot(df_historical['year'], scipy_line_1958, color='blue', label='Trend-line (1958-2024)')
        plot_tl_1958_future = col1_3.checkbox('Climate factor 1.2')
        if plot_tl_1958_future:
            ax.plot([df_historical['year'].iloc[-1], 2099], [scipy_line_1958.iloc[-1], scipy_line_1958_future], '--',
                    color='blue', label='Trend-line (1958-2024) with CF = 1.2')

    if plot_tl_2000:
        plt.plot(df_historical['year'].iloc[42:], scipy_line_2000, color='green', label='Trend-line (2000-2024)')
        plot_tl_2000_future = col1_4.checkbox('Climate factor 1.2 ')
        if plot_tl_2000_future:
            ax.plot([df_historical['year'].iloc[-1], 2099], [scipy_line_2000.iloc[-1], scipy_line_2000_future], '--',
                    color='green', label='Trend-line (2000-2024) with CF = 1.2')
    if plot_45:
        ax.fill_between(future_period_1, rcp45_proj_1[0], rcp45_proj_1[2], color=rcp45_color, alpha=0.2)
        ax.fill_between(future_period_2, rcp45_proj_2[0], rcp45_proj_2[2], color=rcp45_color, alpha=0.2)
        ax.plot(future_period_1, rcp45_proj_1[1], '--', color=rcp45_color, linewidth=2, label='RCP4.5 mid')
        ax.plot(future_period_2, rcp45_proj_2[1], '--', color=rcp45_color, linewidth=2)

    if plot_85:
        ax.fill_between(future_period_1, rcp85_proj_1[0], rcp85_proj_1[2], color=rcp85_color, alpha=0.2)
        ax.fill_between(future_period_2, rcp85_proj_2[0], rcp85_proj_2[2], color=rcp85_color, alpha=0.2)
        ax.plot(future_period_1, rcp85_proj_1[1], '--', color=rcp85_color, linewidth=2, label='RCP8.5 mid')
        ax.plot(future_period_2, rcp85_proj_2[1], '--', color=rcp85_color, linewidth=2)

    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Rainfall [mm]')
    plt.title('Historical and Future Rainfall Projection (RCP Scenarios and climate factor)')
    ax.legend(loc='upper left')
    ax.grid()
    st.pyplot(fig)

    st.write('## Predicted future rainfall in 2100:')
    col2_1, col2_2, col2_3 = st.columns(3)
    if plot_45:
        col2_1.write(f'RCP 4.5 max: {rcp45_proj_2[2][-1]:.0f} mm')
        col2_1.write(f'RCP 4.5 mid: {rcp45_proj_2[1][-1]:.0f} mm')
        col2_1.write(f'RCP 4.5 min: {rcp45_proj_2[0][-1]:.0f} mm')
    if plot_85:
        col2_2.write(f'RCP 8.5 max: {rcp85_proj_2[2][-1]:.0f} mm')
        col2_2.write(f'RCP 8.5 mid: {rcp85_proj_2[1][-1]:.0f} mm')
        col2_2.write(f'RCP 8.5 min: {rcp85_proj_2[0][-1]:.0f} mm')
    if plot_tl_1958:
        if plot_tl_1958_future:
            col2_3.write(f'1958-2024: {scipy_line_1958_future:.0f} mm')
    if plot_tl_2000:
        if plot_tl_2000_future:
            col2_3.write(f'2000-2024: {scipy_line_2000_future:.0f} mm')


# Main code for the DWF and WWF analysis
def flow(rain_file):
    historical_rain = get_file(rain_file[3], RAIN_DIR)
    average_rain_aalesund, average_rain_mr, df_rain_summary = rain_alesund_v_mr(historical_rain)

    num_stations = connection_matrix().shape[0]
    station = list(station_name().keys())[:num_stations]

    G = nx.DiGraph()
    for i in range(num_stations):
        for j in range(num_stations):
            if connection_matrix()[i, j] == 1:
                G.add_edge(station[i], station[j])

    components = list(nx.weakly_connected_components(G))
    end_nodes = {node for node in G.nodes if G.out_degree(node) == 0}

    plot_network = st.radio('',['Network 1', 'Network 2', 'Network 3'],horizontal=True)
    st.sidebar.write('Savgol filter parameter:')
    win = st.sidebar.slider('Window size', min_value=25, max_value=1000, step=25)
    poly = st.sidebar.slider('Poly fit', min_value=0, max_value=8, step=1)

    sorted_components = [sorted(component) for component in components]

    if plot_network == 'Network 1':
        network_plot(sorted_components[:2], G, end_nodes)
        st.header('Dry weather flow vs wet weather flow')
        col1_1, col1_2 = st.columns(2)
        box1 = col1_1.checkbox('Year 2023', False)
        box2 = col1_2.checkbox('Year 2024', False)

        if box1 and plot_network == 'Network 1':
            col1_1.write(list(sorted_components[0]))

            for i in list(sorted_components[0]):
                df_copy = my_savgol_filter(df=get_file(i,DATA_DIR), win=win, poly=poly)
                df_name = i.replace('.parquet', '')
                aligned_data, aligned_data_rain = rain_flow(df_copy)
                wwf_percent, dwf_percent = percent_rain(aligned_data, aligned_data_rain)
                st.write(f'Inflow and infiltration increases the average flow by {wwf_percent - dwf_percent:.1f}%')
                if (wwf_percent-dwf_percent) < 0:
                    plt_num = 3
                else:
                    plt_num = 2
                dwf_wwf_plot(aligned_data, aligned_data_rain, df_name, plt_num, None)

        if box2 and plot_network == 'Network 1':
            col1_2.write(list(sorted_components[1]))
            for i in list(sorted_components[1]):
                df_copy = my_savgol_filter(df=get_file(i,DATA_DIR), win=win, poly=poly)
                df_name = i.replace('.parquet', '')
                aligned_data, aligned_data_rain = rain_flow(df_copy)
                wwf_percent, dwf_percent = percent_rain(aligned_data, aligned_data_rain)
                st.write(f'Inflow and infiltration increases the average flow by {wwf_percent - dwf_percent:.1f}%')
                dwf_wwf_plot(aligned_data, aligned_data_rain, df_name, 2, None)

        st.header('Flow and rain prediction for 2100')
        st.write('The predicted flow in 2100 is based on the average flow from 2023 and 2024 for each station.'
                 'Measurements from the county and the city has been compared to estimate measurement differance for the stations.')
        year_100 = st.checkbox('Prediction for 2100')
        if year_100:
            st.table(df_rain_summary.T.style.format('{:.2f}'))

            future_rain(sorted_components[:2], win, poly, average_rain_aalesund, average_rain_mr)

    elif plot_network == 'Network 2':
        network_plot(sorted_components[2:4], G, end_nodes)
        st.header('Dry weather flow vs wet weather flow')
        col1_1, col1_2 = st.columns(2)
        box1 = col1_1.checkbox('Year 2023', False)
        box2 = col1_2.checkbox('Year 2024', False)

        if box1 and plot_network == 'Network 2':
            col1_1.write(list(components[2]))
            for i in list(sorted_components[2]):
                df_copy = my_savgol_filter(df=get_file(i,DATA_DIR), win=win, poly=poly)
                df_name = i.replace('.parquet', '')
                aligned_data, aligned_data_rain = rain_flow(df_copy)
                wwf_percent, dwf_percent = percent_rain(aligned_data, aligned_data_rain)
                st.write(f'Inflow and infiltration increases the average flow by {wwf_percent - dwf_percent:.1f}%')
                dwf_wwf_plot(aligned_data, aligned_data_rain, df_name, 2, None)

        if box2 and plot_network == 'Network 2':
            col1_2.write(list(sorted_components[3]))
            for i in list(sorted_components[3]):
                df_copy = my_savgol_filter(df=get_file(i,DATA_DIR), win=win, poly=poly)
                df_name = i.replace('.parquet', '')
                aligned_data, aligned_data_rain = rain_flow(df_copy)
                wwf_percent, dwf_percent = percent_rain(aligned_data, aligned_data_rain)
                st.write(f'Inflow and infiltration increases the average flow by {wwf_percent - dwf_percent:.1f}%')
                dwf_wwf_plot(aligned_data, aligned_data_rain, df_name, 2, None)

        st.header('Flow and rain prediction for 2100')
        st.write('The predicted flow in 2100 is based on the average flow from 2023 and 2024 for each station.'
                 'Measurements from the county and the city has been compared to estimate measurement differance for the stations.')
        year_100 = st.checkbox('Prediction for 2100')
        if year_100:
            st.table(df_rain_summary.T.style.format('{:.2f}'))
            future_rain(sorted_components[2:4], win, poly, average_rain_aalesund, average_rain_mr)

    elif plot_network == 'Network 3':
        network_plot(sorted_components[4:6],G,end_nodes)
        st.header('Dry weather flow vs wet weather flow')
        col1_1, col1_2 = st.columns(2)

        box1 = col1_1.checkbox('Year 2023', False)
        box2 = col1_2.checkbox('Year 2024', False)

        if box1 and plot_network == 'Network 3':
            col1_1.write(list(sorted_components[4]))
            for i in list(sorted_components[4]):
                df_copy = my_savgol_filter(df=get_file(i,DATA_DIR), win=win, poly=poly)
                df_name = i.replace('.parquet', '')
                aligned_data, aligned_data_rain = rain_flow(df_copy)
                wwf_percent, dwf_percent = percent_rain(aligned_data, aligned_data_rain)
                st.write(f'Inflow and infiltration increases the average flow by {wwf_percent - dwf_percent:.1f}%')
                dwf_wwf_plot(aligned_data, aligned_data_rain, df_name, 2, None)

        if box2 and plot_network == 'Network 3':
            col1_2.write(list(sorted_components[5]))
            for i in list(sorted_components[5]):
                if i == 'combined_206A_24.parquet':
                    df_copy = my_savgol_filter(df=get_file(i,DATA_DIR), win=win, poly=poly)
                    df_name = i.replace('.parquet', '')
                    aligned_data, aligned_data_rain = rain_flow(df_copy)
                    wwf_percent, dwf_percent = percent_rain(aligned_data, aligned_data_rain)
                    st.write(f'Inflow and infiltration increases the average flow by {wwf_percent - dwf_percent:.1f}%')
                    dwf_wwf_plot(aligned_data, aligned_data_rain, df_name, 2, None)
                if i == 'combined_206_24.parquet':
                    st.write('The dataset for PA206 2024 has a total 19.2% nan-values, so the average is only based on the available data')
                    df_copy = my_savgol_filter(df=get_file(i,DATA_DIR), win=win, poly=poly)
                    df_name = i.replace('.parquet', '')
                    aligned_data, aligned_data_rain = rain_flow(df_copy.iloc[:425927])
                    wwf_percent, dwf_percent = percent_rain(aligned_data, aligned_data_rain)
                    st.write(f'Inflow and infiltration increases the average flow by {wwf_percent - dwf_percent:.1f}%')
                    dwf_wwf_plot(aligned_data, aligned_data_rain, df_name, 2, None)

        st.header('Flow and rain prediction for 2100')
        st.write('The predicted flow in 2100 is based on the average flow from 2023 and 2024 for each station.'
                 'Measurements from the county and the city has been compared to estimate measurement differance for the stations.')
        year_100 = st.checkbox('Prediction for 2100')
        if year_100:
            st.table(df_rain_summary.T.style.format('{:.2f}'))

            future_rain(sorted_components[4:6], win, poly, average_rain_aalesund, average_rain_mr)


# Main code for the control analysis
def control(files):
    st.header('Control')
    selected_file = st.selectbox('Choose available file', files)

    if selected_file:
        df = get_file(selected_file,DATA_DIR)

        st.write('### Preview of', selected_file)
        begin_date = df['timestamp'].iloc[0]
        begin_date2 = begin_date + pd.Timedelta(days=1)
        end_date = df['timestamp'].iloc[-1]

        col1_1, col1_2 = st.columns(2)
        start_1 = col1_1.date_input('Start date',value=begin_date, min_value=begin_date, max_value=end_date)
        end_1 = col1_2.date_input('End date',value=begin_date2, min_value=begin_date, max_value=end_date)

        start_1 = pd.Timestamp(start_1).tz_localize('Europe/Oslo')
        end_1 = pd.Timestamp(end_1).tz_localize('Europe/Oslo')
        chosen_data = df[(df['timestamp'] >= start_1) & (df['timestamp'] < end_1)]

        col2_1, col2_2, col2_3, col2_4 = st.columns(4)
        plot_level = col2_1.checkbox('Raw level data')
        plot_rain = col2_2.checkbox('Rain status')
        plot_flow = col2_3.checkbox('Raw station flow')
        plot_savflow = col2_4.checkbox('Savgol Filter')

        fig, ax = plt.subplots(figsize=(14,8))
        chosen_time = chosen_data[chosen_data['controller_state'] == 'OFF']['timestamp']
        chosen_flow = chosen_data[chosen_data['controller_state'] == 'OFF']['flow_ls']

        if plot_level:
            ax.plot(chosen_data['timestamp'],chosen_data['avg'], alpha=0.9, color='gray')

        if plot_rain:
            ax.plot(chosen_data['timestamp'],chosen_data['rain_status'], color='k')

        if plot_flow:
            ax.scatter(x=chosen_time, y=chosen_flow, c='orange',linewidth=1)

        if plot_savflow:
            st.sidebar.write('Savgol filter parameter:')
            win = st.sidebar.slider('Window size', min_value=25, max_value=1000, step=25)
            poly = st.sidebar.slider('Poly fit', min_value=0, max_value=8, step=1)

            df_copy = my_savgol_filter(df=df, win=win, poly=poly)
            chosen_data2 = df_copy[(df_copy['timestamp'] >= start_1) & (df_copy['timestamp'] < end_1)]

            ax.plot(chosen_data2['timestamp'],chosen_data2['sav_flow'], color='r')

        ax.set_ylabel('Tank level [m] / Flow [l/s]')
        ax.set_xlabel('Timestamp')
        st.pyplot(fig)

        st.write('Further analysis is based on a Savgol filter, please activate "Savgol Filter" to continue.')
        if plot_savflow:
            st.write('### Theoretical control analysis')
            choice = st.selectbox('Method of analysis', ('Default','User defined'))
            val = round(df['avg'].max(), 2)
            col3_1, col3_2, col3_3 = st.columns(3)

            if choice == 'Default':
                col3_1.write('Tank start level = 0m')
                col3_2.write('Pump ON / PID-setpoint = 1.2m')
                col3_3.write('Pump OFF = 0.5m')

                init_tank_level = 0
                pump_setpoint = 1.2
                pump_off = 0.5

            if choice == 'User defined':
                col3_1.write(f'Highest tank level is: {val} ')
                col3_2.write(f'Pump setpoint: 0.20 < x < {val} ')
                col3_3.write(f'Pump OFF: 0.10 < x < setpoint ')

                init_tank_level = col3_1.number_input('Set start tank level', min_value=0.0, max_value=val,step=0.1)
                pump_setpoint = col3_2.number_input('Pump setpoint', min_value=0.2, max_value=val, step=0.1)
                pump_off = col3_3.number_input('Pump OFF', min_value=0.1, max_value=val, step=0.1)

            control_analysis = st.checkbox('Do control analysis', False)

            if control_analysis:
                pump_info = PumpSpecification(selected_file).get_pump_info()
                st.table(pump_info)

                # control_analyse(chosen_data2, init_tank_level, pump_setpoint, pump_off, pump_info)
                user_control = Control(chosen_data2, init_tank_level, pump_setpoint, pump_off, pump_info)
                user_control.consumption_print()
                user_control.plot_control()

                st.write('### Comparing power consumption for average DWF and WWF')
                aligned_data, aligned_data_rain = rain_flow(df_copy)
                if selected_file == 'combined_202A_23.parquet':
                    plot = 3
                else:
                    plot = 2
                dwf_wwf_plot(aligned_data, aligned_data_rain, selected_file, plot, None)

                y = np.linspace(0, 24, 1440)
                aligned_data['timestamp'] = y
                aligned_data_rain['timestamp'] = y
                st.subheader('Power consumption for average DWF')
                dwf_control = Control(aligned_data, init_tank_level, pump_setpoint, pump_off, pump_info)
                dwf_control.consumption_print()
                dwf_control.plot_control()

                st.subheader('Power consumption for average WWF')
                wwf_control = Control(aligned_data_rain, init_tank_level, pump_setpoint, pump_off, pump_info)
                wwf_control.consumption_print()
                wwf_control.plot_control()

                st.write('### Power consumption for scenarios in 2100')
                df_future_rain = compute_and_plot_rain_scenarios(aligned_data, aligned_data_rain, selected_file, 1726.80, 1889.60)
                df_future_rain['timestamp'] = y
                future_scenario = ['RCP 4.5 min', 'RCP 4.5 mid','RCP 4.5 max','RCP 8.5 min', 'RCP 8.5 mid','RCP 8.5 max', '1958-2024 trend','2000-2024 trend']
                selected_scenario = st.selectbox('Choose scenario prediction', future_scenario)

                if selected_scenario == 'RCP 4.5 min':
                    df_future_rain = df_future_rain.rename(columns={'min_45': 'sav_flow'})
                    dwf_wwf_plot(aligned_data, df_future_rain[['sav_flow']], selected_file, plot, None)
                    sen1 = Control(df_future_rain[['timestamp','sav_flow']], init_tank_level, pump_setpoint, pump_off, pump_info)
                    sen1.consumption_print()
                    sen1.plot_control()

                if selected_scenario == 'RCP 4.5 mid':
                    df_future_rain = df_future_rain.rename(columns={'mid_45': 'sav_flow'})
                    dwf_wwf_plot(aligned_data, df_future_rain[['sav_flow']], selected_file, plot, None)
                    sen2 = Control(df_future_rain[['timestamp','sav_flow']], init_tank_level, pump_setpoint, pump_off, pump_info)
                    sen2.consumption_print()
                    sen2.plot_control()

                if selected_scenario == 'RCP 4.5 max':
                    df_future_rain = df_future_rain.rename(columns={'max_45': 'sav_flow'})
                    dwf_wwf_plot(aligned_data, df_future_rain[['sav_flow']], selected_file, plot, None)
                    sen3 = Control(df_future_rain[['timestamp','sav_flow']], init_tank_level, pump_setpoint, pump_off, pump_info)
                    sen3.consumption_print()
                    sen3.plot_control()

                if selected_scenario == 'RCP 8.5 min':
                    df_future_rain = df_future_rain.rename(columns={'min_85': 'sav_flow'})
                    dwf_wwf_plot(aligned_data, df_future_rain[['sav_flow']], selected_file, plot, None)
                    sen4 = Control(df_future_rain[['timestamp','sav_flow']], init_tank_level, pump_setpoint, pump_off, pump_info)
                    sen4.consumption_print()
                    sen4.plot_control()

                if selected_scenario == 'RCP 8.5 mid':
                    df_future_rain = df_future_rain.rename(columns={'mid_85': 'sav_flow'})
                    dwf_wwf_plot(aligned_data, df_future_rain[['sav_flow']], selected_file, plot, None)
                    sen5 = Control(df_future_rain[['timestamp','sav_flow']], init_tank_level, pump_setpoint, pump_off, pump_info)
                    sen5.consumption_print()
                    sen5.plot_control()

                if selected_scenario == 'RCP 8.5 max':
                    df_future_rain = df_future_rain.rename(columns={'max_85': 'sav_flow'})
                    dwf_wwf_plot(aligned_data, df_future_rain[['sav_flow']], selected_file, plot, None)
                    sen6 = Control(df_future_rain[['timestamp','sav_flow']], init_tank_level, pump_setpoint, pump_off, pump_info)
                    sen6.consumption_print()
                    sen6.plot_control()

                if selected_scenario == '1958-2024 trend':
                    df_future_rain = df_future_rain.rename(columns={'hist_1': 'sav_flow'})
                    dwf_wwf_plot(aligned_data, df_future_rain[['sav_flow']], selected_file, plot, None)
                    sen7 = Control(df_future_rain[['timestamp','sav_flow']], init_tank_level, pump_setpoint, pump_off, pump_info)
                    sen7.consumption_print()
                    sen7.plot_control()

                if selected_scenario == '2000-2024 trend':
                    df_future_rain = df_future_rain.rename(columns={'hist_2': 'sav_flow'})
                    dwf_wwf_plot(aligned_data, df_future_rain[['sav_flow']], selected_file, plot, None)
                    sen8 = Control(df_future_rain[['timestamp','sav_flow']], init_tank_level, pump_setpoint, pump_off, pump_info)
                    sen8.consumption_print()
                    sen8.plot_control()

# Main code for the maintenance analysis
def maintenance():
    st.header('Maintenance')
    st.write('This part will not be covered, and moved to future work.')


# Estimated power concumption and cost pr/day for the pumps in the station
def consumption():
    st.header('Station consumption')
    st.write('min power consumption at win=925 and poly=6')
    st.write('max power consumption at win=700 and poly=6')

    power_files = get_available_files(COST_DIR, '.csv')
    power_future_files = get_available_files(COST_DIR_FUTURE, '.csv')

    power_data = power_files[:18]
    avg_flow_power = []
    dwf_flow_power = []
    wwf_flow_power = []

    for f in power_data:
        df = get_file(f,COST_DIR)
        if f.endswith('_23.csv'):
            price = 0.5485
        elif f.endswith('_24.csv'):
            price = 0.4086
        else:
            price = 0

        f = f.replace('poly_win', 'PA')
        f = f.replace('.csv', '')
        avg_flow_power.append({
            'Station': f,
            'Control': 'ON/OFF <br> PID',
            'Min power': f'{df["on_off_c"].iloc[0]:.2f} <br> {df["pid_c"].iloc[0]:.2f}',
            'Max power': f'{df["on_off_c"].iloc[1]:.2f} <br> {df["pid_c"].iloc[1]:.2f}',
            'Min cost': f'{(df["on_off_c"].iloc[0] * price):.2f} <br> {(df["pid_c"].iloc[0] * price):.2f}',
            'Max cost': f'{(df["on_off_c"].iloc[1] * price):.2f} <br> {(df["pid_c"].iloc[1] * price):.2f}',
        })
        dwf_flow_power.append({
            'Station': f,
            'Control': 'ON/OFF <br> PID',
            'Min power': f'{df["on_off_c"].iloc[2]:.2f} <br> {df["pid_c"].iloc[2]:.2f}',
            'Max power': f'{df["on_off_c"].iloc[3]:.2f} <br> {df["pid_c"].iloc[3]:.2f}',
            'Min cost': f'{(df["on_off_c"].iloc[2] * price):.2f} <br> {(df["pid_c"].iloc[2] * price):.2f}',
            'Max cost': f'{(df["on_off_c"].iloc[3] * price):.2f} <br> {(df["pid_c"].iloc[3] * price):.2f}',
        })
        wwf_flow_power.append({
            'Station': f,
            'Control': 'ON/OFF <br> PID',
            'Min power': f'{df["on_off_c"].iloc[4]:.2f} <br> {df["pid_c"].iloc[4]:.2f}',
            'Max power': f'{df["on_off_c"].iloc[5]:.2f} <br> {df["pid_c"].iloc[5]:.2f}',
            'Min cost': f'{(df["on_off_c"].iloc[4] * price):.2f} <br> {(df["pid_c"].iloc[4] * price):.2f}',
            'Max cost': f'{(df["on_off_c"].iloc[5] * price):.2f} <br> {(df["pid_c"].iloc[5] * price):.2f}',
        })

    avg_flow_power = pd.DataFrame(avg_flow_power)
    dwf_flow_power = pd.DataFrame(dwf_flow_power)
    wwf_flow_power = pd.DataFrame(wwf_flow_power)

    st.subheader('Power consumption for daily average flow')
    st.markdown(avg_flow_power.style.hide(axis='index').to_html(escape=False), unsafe_allow_html=True)
    st.subheader('Power consumption for daily DWF flow')
    st.markdown(dwf_flow_power.style.hide(axis='index').to_html(escape=False), unsafe_allow_html=True)
    st.subheader('Power consumption for daily WWF flow')
    st.markdown(wwf_flow_power.style.hide(axis='index').to_html(escape=False), unsafe_allow_html=True)

    for f2 in power_future_files:
        df = get_file(f2, COST_DIR_FUTURE)
        price = (0.5485 + 0.4086)/2
        f2 = f2.replace('combined', 'PA')
        f2 = f2.replace('_23.csv', '')
        avg_future_power = {
            'Scenario': ['RCP 4.5 max',
                         'RCP 4.5 mid',
                         'RCP 4.5 min',
                         'RCP 8.5 max',
                         'RCP 8.5 mid',
                         'RCP 8.5 min',
                         'hist 1',
                         'hist 2',],
            'Control': 'ON/OFF <br> PID',
            'Min power': [f'{df["on_off_c"].iloc[0]:.2f} <br> {df["pid_c"].iloc[0]:.2f}',
                          f'{df["on_off_c"].iloc[1]:.2f} <br> {df["pid_c"].iloc[1]:.2f}',
                          f'{df["on_off_c"].iloc[2]:.2f} <br> {df["pid_c"].iloc[2]:.2f}',
                          f'{df["on_off_c"].iloc[3]:.2f} <br> {df["pid_c"].iloc[3]:.2f}',
                          f'{df["on_off_c"].iloc[4]:.2f} <br> {df["pid_c"].iloc[4]:.2f}',
                          f'{df["on_off_c"].iloc[5]:.2f} <br> {df["pid_c"].iloc[5]:.2f}',
                          f'{df["on_off_c"].iloc[6]:.2f} <br> {df["pid_c"].iloc[6]:.2f}',
                          f'{df["on_off_c"].iloc[7]:.2f} <br> {df["pid_c"].iloc[7]:.2f}',
                          ],

            'Max power': [f'{df["on_off_c"].iloc[8]:.2f} <br> {df["pid_c"].iloc[8]:.2f}',
                          f'{df["on_off_c"].iloc[9]:.2f} <br> {df["pid_c"].iloc[9]:.2f}',
                          f'{df["on_off_c"].iloc[10]:.2f} <br> {df["pid_c"].iloc[10]:.2f}',
                          f'{df["on_off_c"].iloc[11]:.2f} <br> {df["pid_c"].iloc[11]:.2f}',
                          f'{df["on_off_c"].iloc[12]:.2f} <br> {df["pid_c"].iloc[12]:.2f}',
                          f'{df["on_off_c"].iloc[13]:.2f} <br> {df["pid_c"].iloc[13]:.2f}',
                          f'{df["on_off_c"].iloc[14]:.2f} <br> {df["pid_c"].iloc[14]:.2f}',
                          f'{df["on_off_c"].iloc[15]:.2f} <br> {df["pid_c"].iloc[15]:.2f}',
                          ],

            'Min cost': [f'{(df["on_off_c"].iloc[0])*price:.2f} <br> {(df["pid_c"].iloc[0])*price:.2f}',
                          f'{(df["on_off_c"].iloc[1])*price:.2f} <br> {(df["pid_c"].iloc[1])*price:.2f}',
                          f'{(df["on_off_c"].iloc[2])*price:.2f} <br> {(df["pid_c"].iloc[2])*price:.2f}',
                          f'{(df["on_off_c"].iloc[3])*price:.2f} <br> {(df["pid_c"].iloc[3])*price:.2f}',
                          f'{(df["on_off_c"].iloc[4])*price:.2f} <br> {(df["pid_c"].iloc[4])*price:.2f}',
                          f'{(df["on_off_c"].iloc[5])*price:.2f} <br> {(df["pid_c"].iloc[5])*price:.2f}',
                          f'{(df["on_off_c"].iloc[6])*price:.2f} <br> {(df["pid_c"].iloc[6])*price:.2f}',
                          f'{(df["on_off_c"].iloc[7])*price:.2f} <br> {(df["pid_c"].iloc[7])*price:.2f}',
                          ],

            'Max cost':  [f'{(df["on_off_c"].iloc[8])*price:.2f} <br> {(df["pid_c"].iloc[8])*price:.2f}',
                          f'{(df["on_off_c"].iloc[9])*price:.2f} <br> {(df["pid_c"].iloc[9])*price:.2f}',
                          f'{(df["on_off_c"].iloc[10])*price:.2f} <br> {(df["pid_c"].iloc[10])*price:.2f}',
                          f'{(df["on_off_c"].iloc[11])*price:.2f} <br> {(df["pid_c"].iloc[11])*price:.2f}',
                          f'{(df["on_off_c"].iloc[12])*price:.2f} <br> {(df["pid_c"].iloc[12])*price:.2f}',
                          f'{(df["on_off_c"].iloc[13])*price:.2f} <br> {(df["pid_c"].iloc[13])*price:.2f}',
                          f'{(df["on_off_c"].iloc[14])*price:.2f} <br> {(df["pid_c"].iloc[14])*price:.2f}',
                          f'{(df["on_off_c"].iloc[15])*price:.2f} <br> {(df["pid_c"].iloc[15])*price:.2f}',
                          ],
        }
        avg_future_power = pd.DataFrame(avg_future_power)
        st.subheader(f'Power consumption scenarios for {f2} in 2100')
        st.markdown(avg_future_power.style.hide(axis='index').to_html(escape=False), unsafe_allow_html=True)


# Main code for the userdefined data
def user_data(rain_file):
    st.header('Userdefined analysis')
    st.write('Dataset has to be from 2023 or 2024, due to availability of railfall-data')
    uploaded_file = st.file_uploader('Upload your level-data here', type='csv')

    if not uploaded_file:
        st.info('Please upload datasets to continue.')
        return

    st.subheader('Add station information:')
    col1_1, col1_2, col1_3 = st.columns(3)
    sump_size = col1_1.number_input('Size of station sump [m^2]', min_value=0.1, max_value=50.0, step=0.1)
    pump_setpoint = col1_2.number_input('Pump setpoint', min_value=0.0, max_value=50.0, step=1.0)
    pump_stop = col1_3.number_input('Pump stop', min_value=0.0, max_value=50.0, step=1.0)

    raw_data = pd.read_csv(uploaded_file)

    st.subheader('Raw data viewer')
    st.dataframe(raw_data.head())

    col1, col2 = st.columns(2)
    time_col = col1.selectbox('Select time column', raw_data.columns)
    level_col = col2.selectbox('Select Level column', raw_data.columns)

    if time_col == level_col:
        st.info('Time and level column cannot be the same')
        return

    raw_data = raw_data.rename(columns={time_col: 'timestamp', level_col: 'level'})
    raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], errors='coerce', utc=True)
    raw_data['timestamp'] = raw_data['timestamp'].dt.tz_convert('Europe/Oslo')
    raw_data_c = raw_data[['timestamp', 'level']].copy()
    col2_1, col2_2 = st.columns(2)
    col2_1.write(raw_data_c.head())
    col2_2.write(raw_data_c.tail())

    # divide years
    df_2023, df_2024 = split_years(raw_data_c)
    if len(df_2023) == 0:
        st.info('No data found for 2023')
        single_period_sim(df_2024,rain_file, 2024, sump_size, pump_setpoint, pump_stop)
    if len(df_2024) == 0:
        st.info('No data found for 2024')
        single_period_sim(df_2023,rain_file, 2023, sump_size, pump_setpoint, pump_stop)
    if len(df_2023) == 0 and len(df_2024) == 0:
        st.info('Analysis cannot be done with data from oustide the year of 2022 and 2024')
        return

    # proceed with 2023 / 2023 / both analysis
    if len(df_2023) > 1 and len(df_2024) > 1:
        nan_values_2023 = df_2023.isna().sum()
        nan_values_2024 = df_2024.isna().sum()

        st.write('Available data for 2023 and 2024')
        col1_2023, col2_2023 = st.columns(2)
        col1_2023.dataframe(df_2023.head())
        col2_2023.dataframe(nan_values_2023)

        col1_2024, col2_2024 = st.columns(2)
        col1_2024.dataframe(df_2024.head())
        col2_2024.dataframe(nan_values_2024)

        # get rain data and convert time
        rain_2023 = prepare_rain_data(rain_file[1])
        rain_2024 = prepare_rain_data(rain_file[2])

        # merge raw data with rain
        df_combined_2023 = merge_with_rain(df_2023, rain_2023)
        df_combined_2024 = merge_with_rain(df_2024, rain_2024)

        win = st.sidebar.slider('Window size', min_value=25, max_value=1000, step=25)
        poly = st.sidebar.slider('Poly fit', min_value=0, max_value=8, step=1)

        data_23, df_copy_23 = process_flow_and_filter(df_combined_2023, sump_size, win, poly)
        data_24, df_copy_24 = process_flow_and_filter(df_combined_2024, sump_size, win, poly)

        data_comb = data_23._append(data_24).reset_index(drop=True)
        df_copy_comb = my_savgol_filter(df=data_comb, win=700, poly=6)

        nan_values_2023_2 = df_copy_23.isna().sum()
        nan_values_2024_2 = df_copy_24.isna().sum()
        nan_values_comb_2 = df_copy_comb.isna().sum()

        st.write(df_copy_23.head())
        st.dataframe(nan_values_2023_2)

        st.write(df_copy_24.head())
        st.dataframe(nan_values_2024_2)

        st.write(df_copy_comb.head())
        st.dataframe(nan_values_comb_2)

        chosen_data_23 = select_day_plot(df_copy_23, 2023)
        chosen_data_24 = select_day_plot(df_copy_24, 2024)

        # DWF WWF
        df_name = 'User_data'
        aligned_data_23, aligned_data_rain_23 = rain_flow(df_copy_23)
        aligned_data_24, aligned_data_rain_24 = rain_flow(df_copy_24)
        aligned_data_comb, aligned_data_rain_comb = rain_flow(df_copy_comb)

        st.subheader('Average flow for 2023')
        wwf_percent_23, dwf_percent_23 = percent_rain(aligned_data_23, aligned_data_rain_23)
        st.write(f'Inflow and infiltration increases the average flow by {wwf_percent_23 - dwf_percent_23:.1f}%')
        if (wwf_percent_23 - dwf_percent_23) > 0:
            dwf_wwf_plot(aligned_data_23, aligned_data_rain_23, df_name, 2, None)
        if (wwf_percent_23 -dwf_percent_23) < 0:
            dwf_wwf_plot(aligned_data_23, aligned_data_rain_23, df_name, 3, None)

        st.subheader('Average flow for 2024')
        wwf_percent_24, dwf_percent_24 = percent_rain(aligned_data_24, aligned_data_rain_24)
        st.write(f'Inflow and infiltration increases the average flow by {wwf_percent_24 - dwf_percent_24:.1f}%')
        if (wwf_percent_24 - dwf_percent_24) > 0:
            dwf_wwf_plot(aligned_data_24, aligned_data_rain_24, df_name, 2, None)
        if (wwf_percent_24 - dwf_percent_24) < 0:
            dwf_wwf_plot(aligned_data_24, aligned_data_rain_24, df_name, 3, None)

        st.subheader('Average flow for combined data')
        wwf_percent_comb, dwf_percent_comb = percent_rain(aligned_data_comb, aligned_data_rain_comb)
        st.write(f'Inflow and infiltration increases the average flow by {wwf_percent_comb - dwf_percent_comb:.1f}%')
        if (wwf_percent_comb - dwf_percent_comb) > 0:
            dwf_wwf_plot(aligned_data_comb, aligned_data_rain_comb, df_name, 2, None)
        if (wwf_percent_comb - dwf_percent_comb) < 0:
            dwf_wwf_plot(aligned_data_comb, aligned_data_rain_comb, df_name, 3, None)

        st.subheader('Pump power consumption')
        col2_1, col2_2, col2_3 = st.columns(3)
        s_current = col2_1.number_input('Pump Start current', min_value=1.0, max_value=2000.0, step=10.0)
        r_current = col2_2.number_input('Pump rated current', min_value=1.0, max_value=2000.0, step=10.0)
        p_f = col2_3.number_input('Pump rated current', min_value=0.01, max_value=1.0, step=0.1)

        pump_data = {
            'Brand': 'Unknown', 'Model-number': 'unknown', 'Configuration': 'unknown', 'Voltage': 230,
            'Start current': s_current, 'Rated current': r_current, 'Wheel size': 'Not specified', 'p_f': p_f,
            'Phase': 3
        }

        # control on daily data and dwf, wwf
        st.subheader('Control for selected date in 2023')
        simulate_control_run(chosen_data_23, pump_setpoint, pump_stop, pump_data)

        st.subheader('Control for selected date in 2024')
        simulate_control_run(chosen_data_24, pump_setpoint, pump_stop, pump_data)

        y = np.linspace(0, 24, 1440)
        aligned_data_23['timestamp'] = y
        aligned_data_24['timestamp'] = y
        aligned_data_comb['timestamp'] = y

        aligned_data_rain_23['timestamp'] = y
        aligned_data_rain_24['timestamp'] = y
        aligned_data_rain_comb['timestamp'] = y

        st.subheader('Power consumption for average DWF, 2023')
        simulate_control_run(aligned_data_23, pump_setpoint, pump_stop, pump_data)
        st.subheader('Power consumption for average WWF, 2023')
        simulate_control_run(aligned_data_rain_23, pump_setpoint, pump_stop, pump_data)

        st.subheader('Power consumption for average DWF, 2024')
        simulate_control_run(aligned_data_24, pump_setpoint, pump_stop, pump_data)
        st.subheader('Power consumption for average WWF, 2024')
        simulate_control_run(aligned_data_rain_24, pump_setpoint, pump_stop, pump_data)

        historical_rain = get_file(rain_file[3], RAIN_DIR)
        average_rain_aalesund, average_rain_mr, df_rain_summary = rain_alesund_v_mr(historical_rain)

        st.subheader('Rainfall scenarios for station in 2100')
        df_future_rain = compute_and_plot_rain_scenarios(aligned_data_comb, aligned_data_rain_comb, df_name, average_rain_aalesund, average_rain_mr)
        dwf_wwf_plot(aligned_data_comb, df_future_rain, df_name, 4, aligned_data_rain_comb)
        st.subheader('Power consumption for WWF during RCP 8.5 max, for year 2100')
        df_future_rain['timestamp'] = y
        df_future_rain = df_future_rain.rename(columns={f'max_85': 'sav_flow'})
        simulate_control_run(df_future_rain[['timestamp','sav_flow']], pump_setpoint, pump_stop, pump_data)


# ------------ Additional functions used in the main code ------------
def historical_rain(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df['year'] = pd.to_datetime(df['date']).dt.year
    year_list = []
    for year, group in df.groupby('year'):
        rain = group['rr'].sum()
        year_list.append((year, rain))

    df_hist = pd.DataFrame(year_list, columns=['year', 'rain'])
    return df_hist


# Smoothing filter for the raw calculated flow
def my_savgol_filter(df, win, poly):
    df_copy = df.copy()
    mask_controller = df_copy['controller_state'] == 'OFF'
    mask_c = df_copy['flow_ls'].where(mask_controller)
    filtered_off_c = np.full_like(mask_c, np.nan)
    valid_indices_c = mask_c.dropna().index
    smoothed_values_c = scipy.signal.savgol_filter(mask_c.dropna(), window_length=win, polyorder=poly, mode='nearest')
    last_positive = 0
    for i in range(len(smoothed_values_c)):
        if smoothed_values_c[i] < 0:
            smoothed_values_c[i] = last_positive
        else:
            last_positive = smoothed_values_c[i]
    filtered_off_c[valid_indices_c] = smoothed_values_c
    df_copy['sav_flow'] = filtered_off_c
    df_copy['sav_flow'] = df_copy['sav_flow'].ffill()
    df_copy['sav_flow'] = df_copy['sav_flow'].bfill()
    return df_copy


# Pump-station network connection
def network_plot(components, G, end_nodes):
    fig, axes = plt.subplots(1,2, figsize=(14, 7))
    axes = axes.flatten()
    for idx, component in enumerate(components):
        sub_G = G.subgraph(component).copy()
        for node in component:
            if node in end_nodes:
                sub_G.add_edge(node, 'WWTP')

        pos = nx.spring_layout(sorted(sub_G), seed=41)   # 8, 41

        ax = axes[idx]
        labels = {node: node.replace('.parquet', '') for node in sub_G.nodes}
        nx.draw(sub_G, pos, with_labels=True, labels=labels, node_color='lightblue', edge_color='gray',
                node_size=2500, font_size=10, ax=ax)

        if 'WWTP' in sub_G:
            nx.draw_networkx_nodes(sub_G, pos, nodelist=['WWTP'], node_color='red', node_size=2500, ax=ax)

        ax.set_title(f'Station data from {idx + 2023}')
    st.pyplot(fig)


# Names of the pump-stations in the analysis
def station_name():
    return {'combined_201_23.parquet': '2023',
            'combined_201_24.parquet': '2024',
            'combined_202A_23.parquet': '2023',
            'combined_202A_24.parquet': '2024',
            'combined_202_23.parquet': '2023',
            'combined_202_24.parquet': '2024',
            'combined_203_23.parquet': '2023',
            'combined_203_24.parquet': '2024',
            'combined_204_23.parquet': '2023',
            'combined_204_24.parquet': '2024',
            'combined_205A_23.parquet': '2023',
            'combined_205A_24.parquet': '2024',
            'combined_205_23.parquet': '2023',
            'combined_205_24.parquet': '2024',
            'combined_206A_23.parquet': '2023',
            'combined_206A_24.parquet': '2024',
            'combined_206_23.parquet': '2023',
            'combined_206_24.parquet': '2024',
            }


# Connection ID for the pump-stations
def connection_matrix():
    return np.array([[-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]])


# Categorize periods with and without rain
def rain_flow(df):
    df_no_rain = df[(df['day_rainfall'] == 0)][['timestamp', 'sav_flow']]
    df_rain = df[(df['day_rainfall'] > 0)][['timestamp', 'sav_flow']]
    df_no_rain['date'] = df_no_rain['timestamp'].dt.date
    df_rain['date'] = df_rain['timestamp'].dt.date

    aligned_data = []
    aligned_data_rain = []

    for date, group in df_no_rain.groupby('date'):
        group = group.reset_index(drop=True)
        aligned_data.append(group['sav_flow'])
    for date, group in df_rain.groupby('date'):
        group = group.reset_index(drop=True)
        aligned_data_rain.append(group['sav_flow'])

    aligned_data1 = pd.DataFrame(aligned_data).T
    aligned_data_rain = pd.DataFrame(aligned_data_rain).T

    aligned_data = aligned_data1.mean(axis=1)
    aligned_data_rain = aligned_data_rain.mean(axis=1)
    aligned_data = pd.DataFrame(aligned_data[:1440], columns=['sav_flow'])
    aligned_data_rain = pd.DataFrame(aligned_data_rain[:1440], columns=['sav_flow'])
    return aligned_data, aligned_data_rain


# Plots for DWF and WWF
def dwf_wwf_plot(df, df_rain, name, plot_num, df_rain_2):
    y = np.linspace(0, 24, 1440)
    if plot_num == 1:
        fig, ax = plt.subplots( figsize=(12, 4))
        plt.suptitle(f'Station data from {name}')
        ax.plot(y, df['sav_flow'], c='blue', label='Avg DWF (24h)')
        ax.plot(y, df_rain['sav_flow'],c='red', label='Average WWF (24h)')
        ax.set_xlabel('Time [hour]')
        ax.set_ylabel('Station flow [l/s]')
        ax.legend()
        st.pyplot(fig)

    if plot_num == 2:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), )
        plt.suptitle(f'Station data from {name}')
        ax[0].plot(y, df['sav_flow'], c='blue', label='Average DWF (24h)')
        ax[0].plot(y, df_rain['sav_flow'], c='r', label='Average WWF (24h)')
        ax[0].set_xlabel('Time [hour]')
        ax[0].set_ylabel('Station flow [l/s]')
        ax[0].legend()

        pie_label = ['Average WWF', 'Average DWF']
        wwf_percent, dwf_percent = percent_rain(df, df_rain)
        ax[1].pie([(wwf_percent-dwf_percent),dwf_percent], colors=['red', 'blue'], autopct='%1.1f%%', wedgeprops=dict(edgecolor='w'))
        ax[1].legend(labels=pie_label, loc='upper left')
        st.pyplot(fig)

    if plot_num == 3:
        fig, ax = plt.subplots(1,2, figsize=(12, 4),)
        plt.suptitle(f'Station data from {name}')
        ax[0].plot(y, df['sav_flow'], c='blue', label='Average DWF (24h)')
        ax[0].plot(y, df_rain['sav_flow'], c='r', label='Average WWF (24h)')
        ax[0].set_xlabel('Time [hour]')
        ax[0].set_ylabel('Station flow [l/s]')
        ax[0].legend()

        wwf_percent, dwf_percent = percent_rain(df,df_rain)
        ax[1].bar('WWF', wwf_percent, color='red', label='Sum of WWF')
        ax[1].bar('DWF', dwf_percent, color='blue', label='Sum of DWF')
        ax[1].set_ylabel('%')
        ax[1].legend()
        st.pyplot(fig)

    if plot_num == 4:
        fig, ax = plt.subplots( figsize=(12, 4))
        plt.suptitle(f'Station data from {name}')
        ax.plot(y, df['sav_flow'], c='blue', label='Avg DWF (24h)')
        ax.plot(y, df_rain_2['sav_flow'], c='purple', label='Avg WWF (24h)')
        # ax.plot(y, df_rain['max_45'], '--', color='orange', label='max_45', linewidth=2)
        ax.plot(y, df_rain['mid_45'], color='orange', label='mid_45', linewidth=3)
        # ax.plot(y, df_rain['min_45'], '--', color='orange', label='min_45', linewidth=2)
        # ax.plot(y, df_rain['max_85'], '--', color='red', label='max_85', linewidth=2)
        ax.plot(y, df_rain['mid_85'], color='red', label='mid_85', linewidth=3)
        # ax.plot(y, df_rain['min_85'], '--', color='red', label='min_85', linewidth=2)
        ax.plot(y, df_rain['hist_1'], color='green', label='**', linewidth=3)
        ax.plot(y, df_rain['hist_2'], '--', color='green', label='***', linewidth=2)
        ax.set_xlabel('Time [hour]')
        ax.set_ylabel('Station flow [l/s]')
        ax.legend(loc='upper right')
        st.pyplot(fig)
        st.write('** Future rainfall based on trend from 1958-2024')
        st.write('*** Future rainfall based on trend from 2000-2024')


# Calculate percent amount which can be considered rain
def percent_rain(df,df_rain):
    wwf_percent = None
    dwf_percent = None
    if df['sav_flow'].sum() < df_rain['sav_flow'].sum():
        wwf_percent = (df_rain['sav_flow'].sum()/df_rain['sav_flow'].sum())*100
        dwf_percent = (df['sav_flow'].sum()/df_rain['sav_flow'].sum())*100
    if df['sav_flow'].sum() > df_rain['sav_flow'].sum():
        dwf_percent = (df['sav_flow'].sum() / df['sav_flow'].sum()) * 100
        wwf_percent = (df_rain['sav_flow'].sum() / df['sav_flow'].sum()) * 100

    return wwf_percent, dwf_percent #, p_rain


# Simple regression of data
def regression(df):
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['year'], df['rain'])
    return slope, intercept


# Future calculation of the regression
def scipy_line_future(slope, intercept, year):
    climate_factor = 1.2
    scipy_line = (intercept + slope * year) * climate_factor
    return scipy_line


# Historical trend of data
def scipy_line_historical(slope, intercept, df):
    scipy_line = intercept + slope * df
    return scipy_line


# Representative Concentration Pathway for 4.5 and 8.5 for Møre og Romsdal
def rcp(future_period_1, future_period_2, baseline):
    def compute_projection(base, years, rates):
        projections = []
        for rate in rates:
            x = base * (1 + rate)
            n = len(years) - 1
            g = (x / base) **(1/n) -1
            proj = base * (1 + g)**(years - years[0])
            projections.append(proj)
        return np.array(projections)

    rcp45_rates_1 = np.array([-0.01, 0.05, 0.11])
    rcp85_rates_1 = np.array([0.01, 0.07, 0.11])

    rcp45_rates_2 = np.array([-0.02, 0.06, 0.10])
    rcp85_rates_2 = np.array([0.02, 0.14, 0.18])

    rcp45_proj_1 = compute_projection(baseline, future_period_1, rcp45_rates_1)
    rcp85_proj_1 = compute_projection(baseline, future_period_1, rcp85_rates_1)

    rcp45_proj_2 = np.array([compute_projection(rcp45_proj_1[i, -1], future_period_2, [rcp45_rates_2[i]])[0] for i in range(3)])
    rcp85_proj_2 = np.array([compute_projection(rcp85_proj_1[i, -1], future_period_2, [rcp85_rates_2[i]])[0] for i in range(3)])
    return rcp45_proj_1, rcp45_proj_2, rcp85_proj_1, rcp85_proj_2


def future_rain(file, win, poly, rain_aalesund, rain_mr):
    average_rain_aalesund = rain_aalesund
    average_rain_MR = rain_mr

    for r in range(len(file[:2][0])):
        file1 = get_file(file[0][r], DATA_DIR)
        file2 = get_file(file[1][r], DATA_DIR)

        if file[1][r] == 'combined_206_24.parquet':
            file2 = file2.iloc[:425927]

        file1 = file1._append(file2).reset_index(drop=True)
        sav_tot = my_savgol_filter(df=file1, win=win, poly=poly)
        aligned_data, aligned_data_rain = rain_flow(sav_tot)
        df_name = file[0][r].replace('_23.parquet', '')

        df_future_rain = compute_and_plot_rain_scenarios(aligned_data, aligned_data_rain, df_name, average_rain_aalesund, average_rain_MR)
        dwf_wwf_plot(aligned_data, df_future_rain, df_name, 4, aligned_data_rain)


def compute_and_plot_rain_scenarios(aligned_data, aligned_data_rain, df_name, average_rain_aalesund, average_rain_MR):
    aalesund_MR_ratio = average_rain_aalesund / average_rain_MR

    rpc_max_45 = 2297 * aalesund_MR_ratio
    rpc_mid_45 = 2094 * aalesund_MR_ratio
    rpc_min_45 = 1826 * aalesund_MR_ratio
    rpc_max_85 = 2465 * aalesund_MR_ratio
    rpc_mid_85 = 2295 * aalesund_MR_ratio
    rpc_min_85 = 1938 * aalesund_MR_ratio
    hist_1958_2024 = 2600 * aalesund_MR_ratio
    hist_2000_2024 = 2371 * aalesund_MR_ratio

    future_rain_scenario = np.zeros((len(aligned_data), 8))

    for i in range(len(aligned_data)):
        step_diff = ((aligned_data_rain['sav_flow'].iloc[i] - aligned_data['sav_flow'].iloc[i]) / aligned_data['sav_flow'].iloc[i])
        scenario_1 = scenario_rain(aligned_data['sav_flow'].iloc[i], step_diff, average_rain_aalesund, rpc_max_45)
        scenario_2 = scenario_rain(aligned_data['sav_flow'].iloc[i], step_diff, average_rain_aalesund, rpc_mid_45)
        scenario_3 = scenario_rain(aligned_data['sav_flow'].iloc[i], step_diff, average_rain_aalesund, rpc_min_45)
        scenario_4 = scenario_rain(aligned_data['sav_flow'].iloc[i], step_diff, average_rain_aalesund, rpc_max_85)
        scenario_5 = scenario_rain(aligned_data['sav_flow'].iloc[i], step_diff, average_rain_aalesund, rpc_mid_85)
        scenario_6 = scenario_rain(aligned_data['sav_flow'].iloc[i], step_diff, average_rain_aalesund, rpc_min_85)
        scenario_7 = scenario_rain(aligned_data['sav_flow'].iloc[i], step_diff, average_rain_aalesund, hist_1958_2024)
        scenario_8 = scenario_rain(aligned_data['sav_flow'].iloc[i], step_diff, average_rain_aalesund, hist_2000_2024)

        future_rain_scenario[i] = [scenario_1, scenario_2, scenario_3, scenario_4, scenario_5, scenario_6, scenario_7, scenario_8]

    df_future_rain = pd.DataFrame(future_rain_scenario, columns=['max_45', 'mid_45', 'min_45', 'max_85', 'mid_85', 'min_85', 'hist_1', 'hist_2'])
    return df_future_rain
    # dwf_wwf_plot(aligned_data, df_future_rain, df_name, 4)


def scenario_rain(df, step, rain, f_rain):
    s_flow = df * (1 + (step * f_rain / rain))
    return s_flow


def rain_alesund_v_mr(df):
    df_historical = historical_rain(df)
    rain_2023 = get_file(rain_files[1], RAIN_DIR)
    rain_2024 = get_file(rain_files[2], RAIN_DIR)
    rain_aalesund_2023 = rain_2023['day_rainfall'].sum()
    rain_aalesund_2024 = rain_2024['day_rainfall'].sum()
    average_rain_aalesund = (rain_aalesund_2023 + rain_aalesund_2024) / 2
    rain_MR_2023 = df_historical[(df_historical['year'] == 2023)]['rain'].iloc[0]
    rain_MR_2024 = df_historical[df_historical['year'] == 2024]['rain'].iloc[0]
    average_rain_MR = (rain_MR_2023 + rain_MR_2024) / 2

    df1 = {
        'location': ['Ålesund', 'Møre og Romsdal'],
        'Total measured rain 2023 [mm]': [rain_aalesund_2023, rain_MR_2023],
        'Total measured rain 2024 [mm]': [rain_aalesund_2024, rain_MR_2024],
        'Average measured rain [mm]': [average_rain_aalesund, average_rain_MR],
    }
    rain_summary = pd.DataFrame(df1)
    rain_summary = rain_summary.set_index('location')
    return average_rain_aalesund, average_rain_MR, rain_summary


# Classification of the pumps with information in the pumpstations
class PumpSpecification:
    def __init__(self, name: str):
        self.name = name
        self.pump_data = {
            'Pump 1': {'Brand': 'Xylem', 'Model-number': '6020.180', 'Configuration': 'Wet', 'Voltage': 230, 'Start current': 6.2, 'Rated current': 6.2, 'Wheel size': 'Not specified', 'p_f': 0.95, 'Phase': 3},
            'Pump 2': {'Brand': 'Xylem', 'Model-number': '3127.160', 'Configuration': 'Wet', 'Voltage': 230, 'Start current': 195, 'Rated current': 24, 'Wheel size': 246, 'p_f': 0.89, 'Phase': 3},
            'Pump 3': {'Brand': 'Xylem', 'Model-number': '3127.160', 'Configuration': 'Dry', 'Voltage': 230, 'Start current': 109, 'Rated current': 21, 'Wheel size': 487, 'p_f': 0.88, 'Phase': 3},
            'Pump 4': {'Brand': 'Xylem', 'Model-number': '3171.181', 'Configuration': 'Wet', 'Voltage': 230, 'Start current': 462, 'Rated current': 67, 'Wheel size': 275, 'p_f': 0.93, 'Phase': 3},
            'Pump 5': {'Brand': 'Xylem', 'Model-number': '3127.180', 'Configuration': 'Wet', 'Voltage': 400, 'Start current': 62, 'Rated current': 12, 'Wheel size': 480, 'p_f': 0.88, 'Phase': 3},
            'Pump 6': {'Brand': 'Xylem', 'Model-number': '3127.170', 'Configuration': 'Wet', 'Voltage': 230, 'Start current': 114, 'Rated current': 24, 'Wheel size': 210, 'p_f': 0.89, 'Phase': 3},
            'Pump 7': {'Brand': 'Xylem', 'Model-number': '3069.170', 'Configuration': 'Wet', 'Voltage': 230, 'Start current': 47, 'Rated current': 8.8, 'Wheel size': 252, 'p_f': 0.86, 'Phase': 3},
            'Pump 8': {'Brand': 'Xylem', 'Model-number': '3102.900', 'Configuration': 'Dry', 'Voltage': 230, 'Start current': 72, 'Rated current': 11, 'Wheel size': 462, 'p_f': 0.88, 'Phase': 3},
            'Pump 9': {'Brand': 'Grundfos', 'Model-number': '96047789', 'Configuration': 'Dry', 'Voltage': 230, 'Start current': 55, 'Rated current': 10.3, 'Wheel size': 'Not specified', 'p_f': 0.74, 'Phase': 3},
        }
        self.station_name = {
            'combined_201_23.parquet': ['Pump 1'],
            'combined_201_24.parquet': ['Pump 1'],
            'combined_202A_23.parquet': ['Pump 3'],
            'combined_202A_24.parquet': ['Pump 3'],
            'combined_202_23.parquet': ['Pump 2'],
            'combined_202_24.parquet': ['Pump 2'],
            'combined_203_23.parquet': ['Pump 4'],
            'combined_203_24.parquet': ['Pump 4'],
            'combined_204_23.parquet': ['Pump 5'],
            'combined_204_24.parquet': ['Pump 5'],
            'combined_205A_23.parquet': ['Pump 7'],
            'combined_205A_24.parquet': ['Pump 7'],
            'combined_205_23.parquet': ['Pump 6'],
            'combined_205_24.parquet': ['Pump 6'],
            'combined_206A_23.parquet': ['Pump 9'],
            'combined_206A_24.parquet': ['Pump 9'],
            'combined_206_23.parquet': ['Pump 8'],
            'combined_206_24.parquet': ['Pump 8'],
            }

    def get_pump_info(self):
        return self.pump_data.get(self.station_name.get(self.name)[0])


class Control:
    def __init__(self, list_data, start_level, setpoint, stop, pump_info):
        self.list_data = list_data.copy()
        self.start_level = start_level
        self.setpoint = setpoint
        self.pump_stop = stop
        self.pump_info = pump_info

        self.previous_error = 0.0
        self.integral = 0.0

        self.W_to_kW = 1000
        self.startup_time = 10

        self.Kp = 0.184
        self.Ki = 0.0055
        self.Kd = 0.005

        self.factor = 1000 * 60 ** (-1)

        self.eff_flow_l = 5 / self.factor

        if isinstance(self.list_data, pd.DataFrame):
            self.list_data['sav_flow'] = self.list_data['sav_flow'] / self.factor
            self.np_flow = self.list_data['sav_flow'].to_numpy()
        elif isinstance(self.list_data, pd.Series):
            self.list_data = self.list_data / self.factor
            self.np_flow = self.list_data.to_numpy()
        elif isinstance(self.list_data, np.ndarray):
            self.list_data = self.list_data / self.factor
            self.np_flow = self.list_data
        else:
            raise TypeError('Unsupported data type for list_data.')

        self.df1 = self.on_off_control()
        self.df2 = self.pid_control()

    def predict_tank_level_rk4(self, h0, r, y, T_pred, dt):
        """Predict future tank level using RK4 method"""
        h_future = h0
        num_steps = int(T_pred / dt)
        pump = self.eff_flow_l
        for _ in range(num_steps):
            k1 = (y - r * pump)
            k2 = (y - r * pump + (dt / 2) * k1)
            k3 = (y - r * pump + (dt / 2) * k2)
            k4 = (y - r * pump + dt * k3)
            h_future += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h_future

    def pid_calculation(self, desired_level, current_level):
        error = desired_level - current_level
        self.integral += error
        derivative = error - self.previous_error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = max(min(output, 1), 0)

        self.previous_error = error
        return output

    def on_off_control(self):
        ON_OFF_data = []
        current_tank_level = self.start_level
        controller_state = 'OFF'

        ON_OFF_data.append((current_tank_level, controller_state))
        for i in range(len(self.np_flow) - 1):
            if current_tank_level < self.pump_stop:
                controller_state = 'OFF'
            elif current_tank_level > self.setpoint:
                controller_state = 'ON'

            if controller_state == 'OFF':
                current_tank_level += self.np_flow[i]
            elif controller_state == 'ON':
                value = self.np_flow[i] - self.eff_flow_l
                current_tank_level += value

            ON_OFF_data.append((current_tank_level, controller_state))

        self.ON_OFF_df = pd.DataFrame(ON_OFF_data)
        self.ON_OFF_df.columns = ['current_tank_level', 'controller_state']
        return self.ON_OFF_df

    def pid_control(self):
        tf = len(self.np_flow)
        t_step = 1.0
        T_pred = 120.0
        t = np.arange(1, tf, t_step)
        results = np.zeros((len(t)+1, 8))

        results[0] = [0, 0, self.setpoint, self.start_level, 0, 0, 0, self.pump_stop]
        for i, ti in enumerate(t):
            y = self.np_flow[i]
            r = self.pid_calculation(self.setpoint, self.start_level)
            h_target = self.predict_tank_level_rk4(self.start_level, r, y, T_pred, t_step)

            blend = h_target - self.start_level
            error_future = self.setpoint - blend

            r = self.pid_calculation(self.setpoint, error_future)

            self.start_level += y - r * self.eff_flow_l

            results[i+1] = [ti, r, self.setpoint, self.start_level, error_future, h_target, y, self.pump_stop]

        self.PID_df = pd.DataFrame(results, columns=['ti', 'r', 'sp', 'h0', 'err_future', 'h_target', 'y', 'off']).set_index('ti')
        return self.PID_df

    def power_consumption(self):
        df1 = self.df1
        df2 = self.df2
        voltage = self.pump_info['Voltage']
        starting_current = self.pump_info['Start current']
        rated_current = self.pump_info['Rated current']
        power_factor = self.pump_info['p_f']
        phase = self.pump_info['Phase']

        on_off_ON_time = df1[df1['controller_state'] == 'ON']['controller_state'].values

        on_off_counter = df1['controller_state'].shift()
        self.off_to_on_counter = ((df1['controller_state'] == 'ON') & (on_off_counter == 'OFF')).sum()

        kW = (np.sqrt(phase) * voltage * rated_current * power_factor) / self.W_to_kW
        kW_start = (((np.sqrt(phase) * voltage * starting_current * (power_factor/2)) / self.W_to_kW) * self.startup_time) / 3600
        kW_start_total = kW_start * self.off_to_on_counter

        pid_time = df2[df2['r'] > 0]['r'].values
        on_off_pid_counter = df2['r'].shift()
        self.pid_on_counter = ((df2['r'] > 0) & (on_off_pid_counter == 0)).sum()

        self.on_off_consumption = (len(on_off_ON_time) * kW / 60) + kW_start_total
        e = 0
        for s in pid_time:
            p = kW * (max(s, 0.4))**3
            e += p * (1/60)
        self.pid_consumption = e + (kW_start * self.pid_on_counter)
        # self.pid_consumption = (pid_time.sum() * kW / 60) + (kW_start * self.pid_on_counter)

        self.on_off_duration = len(on_off_ON_time)
        self.pid_duration = len(pid_time)

        return self.off_to_on_counter, self.pid_on_counter, self.on_off_consumption, self.pid_consumption, self.on_off_duration, self.pid_duration

    def plot_control(self):
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(self.list_data['timestamp'], self.df1['current_tank_level'], c='darkorange', label='Tank Level - ON/OFF')
        ax.plot(self.list_data['timestamp'], self.df2['r'], 'r--', label='PID Output (r)')
        ax.plot(self.list_data['timestamp'], self.df2['h0'], c='purple', label='Tank Level - PID')
        ax.plot(self.list_data['timestamp'], self.df2['sp'], 'k:', label='Setpoint')
        ax.plot(self.list_data['timestamp'], self.df2['off'], 'k:')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Tank level [m]')
        ax.legend(loc='center left')
        st.pyplot(fig)

    def consumption_print(self):
        self.power_consumption()
        st.write(f'The pump starts {self.off_to_on_counter} times in the simulation for the ON-OFF controller')
        st.write(f'ON-OFF consumes: {self.on_off_consumption:.3f} kWh and runs for {self.on_off_duration} minutes')
        st.write('')
        st.write(f'The pump starts {self.pid_on_counter} times in the simulation for the PID controller')
        st.write(f'PID consumes: {self.pid_consumption:.3f} kWh and runs for {self.pid_duration} minutes')


def time_convert(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('Europe/Oslo').dt.date
    return df


def calculate_flow(df, sump_diam):
    time_deltas = df['timestamp'].diff()
    timestep = time_deltas.mode()[0]
    x = timestep.total_seconds()
    # tank_a = sump_diam
    tank_a = 1

    m_to_l = 1000
    min_to_sec = x ** (-1)
    flow_ls = 0

    controller_state = None
    current_flow_state = []
    for i in range(len(df)):
        if i < len(df) - 1:
            flow_m = df['level'].iloc[i + 1] - df['level'].iloc[i]
            if flow_m < 0:
                flow_ls = flow_m * tank_a * m_to_l * min_to_sec
                controller_state = 'ON'
            else:
                flow_ls = flow_m * tank_a * m_to_l * min_to_sec
                controller_state = 'OFF'
        else:
            pass

        current_flow_state.append((i + 1, flow_ls, controller_state))

    station_flow = pd.DataFrame(current_flow_state, columns=['step', 'flow_ls', 'controller_state'])
    df_combined = df.merge(station_flow[['flow_ls', 'controller_state']], left_index=True, right_index=True, how='left')
    return df_combined


def single_period_sim(df, rain_file, year, sump_size, pump_setpoint, pump_stop):
    nan_values = df.isna().sum()

    st.write(f'Available data for {year}')
    col1, col2 = st.columns(2)
    col1.dataframe(df.head())
    col2.dataframe(nan_values)

    # get rain data and convert time
    if year == 2023: # file 3 = 2023, file 4 = 2024
        rain = prepare_rain_data(rain_file[1])
    if year == 2024:
        rain = prepare_rain_data(rain_file[2])

    # merge raw data with rain
    df_combined = merge_with_rain(df, rain)

    win = st.sidebar.slider('Window size', min_value=25, max_value=1000, step=25)
    poly = st.sidebar.slider('Poly fit', min_value=0, max_value=8, step=1)

    data, df_copy = process_flow_and_filter(df_combined, sump_size, win, poly)

    data_comb = data
    df_copy_comb = my_savgol_filter(df=data_comb, win=700, poly=6)

    nan_values_comb_2 = df_copy_comb.isna().sum()

    st.write(df_copy_comb.head())
    st.dataframe(nan_values_comb_2)

    chosen_data = select_day_plot(df_copy, 2023)

    # DWF WWF
    df_name = f'data from {year}'
    aligned_data, aligned_data_rain = rain_flow(df_copy)
    aligned_data_comb, aligned_data_rain_comb = rain_flow(df_copy_comb)

    st.subheader(f'Average flow for {year}')
    wwf_percent, dwf_percent = percent_rain(aligned_data, aligned_data_rain)
    st.write(f'Inflow and infiltration increases the average flow by {wwf_percent - dwf_percent:.1f}%')
    if (wwf_percent - dwf_percent) > 0:
        dwf_wwf_plot(aligned_data, aligned_data_rain, df_name, 2, None)
    if (wwf_percent -dwf_percent) < 0:
        dwf_wwf_plot(aligned_data, aligned_data_rain, df_name, 3, None)

    st.subheader('Pump power consumption')
    col2_1, col2_2, col2_3 = st.columns(3)
    s_current = col2_1.number_input('Pump Start current', min_value=1.0, max_value=2000.0, step=10.0)
    r_current = col2_2.number_input('Pump rated current', min_value=1.0, max_value=2000.0, step=10.0)
    p_f = col2_3.number_input('Pump rated current', min_value=0.01, max_value=1.0, step=0.1)

    pump_data = {
        'Brand': 'Unknown', 'Model-number': 'unknown', 'Configuration': 'unknown', 'Voltage': 230, 'Start current': s_current, 'Rated current': r_current, 'Wheel size': 'Not specified', 'p_f': p_f, 'Phase': 3
    }

    # control on daily data and dwf, wwf
    st.subheader(f'Control for selected date in {year}')
    simulate_control_run(chosen_data, pump_setpoint, pump_stop, pump_data)

    y = np.linspace(0, 24, 1440)
    aligned_data['timestamp'] = y
    aligned_data_comb['timestamp'] = y

    aligned_data_rain['timestamp'] = y
    aligned_data_rain_comb['timestamp'] = y

    st.subheader(f'Power consumption for average DWF, {year}')
    simulate_control_run(aligned_data, pump_setpoint, pump_stop, pump_data)
    st.subheader(f'Power consumption for average WWF, {year}')
    simulate_control_run(aligned_data_rain, pump_setpoint, pump_stop, pump_data)

    historical_rain = get_file(rain_file[3], RAIN_DIR)
    average_rain_aalesund, average_rain_mr, df_rain_summary = rain_alesund_v_mr(historical_rain)

    st.subheader('Rainfall scenarios for station in 2100')
    df_future_rain = compute_and_plot_rain_scenarios(aligned_data_comb, aligned_data_rain_comb, df_name, average_rain_aalesund, average_rain_mr)
    dwf_wwf_plot(aligned_data_comb, df_future_rain, df_name, 4, aligned_data_rain_comb)
    st.subheader('Power consumption for WWF during RCP 8.5 max, for year 2100')
    df_future_rain['timestamp'] = y
    df_future_rain = df_future_rain.rename(columns={f'max_85': 'sav_flow'})
    simulate_control_run(df_future_rain[['timestamp','sav_flow']], pump_setpoint, pump_stop, pump_data)


@st.cache_data(show_spinner='Splitting data...')
def split_years(df):
    df['year'] = df['timestamp'].dt.year
    df_2023 = df[df['year'] == 2023].drop(columns='year').reset_index(drop=True)
    df_2024 = df[df['year'] == 2024].drop(columns='year').reset_index(drop=True)
    return df_2023, df_2024


@st.cache_data(show_spinner='Loading rain data...')
def prepare_rain_data(rain_path):
    df_rain = get_file(rain_path, RAIN_DIR)
    df_rain['timestamp'] = pd.to_datetime(df_rain['timestamp'], errors='coerce', utc=True)
    return df_rain


@st.cache_data(show_spinner='Merging with rainfall data...')
def merge_with_rain(df, rain_df):
    df_merged = df.merge(rain_df[['timestamp', 'day_rainfall']], on='timestamp', how='left')
    df_merged = df_merged.ffill()
    df_merged = df_merged.bfill()
    df_merged['rain_status'] = df_merged['day_rainfall'] > 0
    return df_merged


# @st.cache_data(show_spinner='Computing data...')
def process_flow_and_filter(df, sump_size, win, poly):
    df_flow = calculate_flow(df, sump_size)
    df_filtered = my_savgol_filter(df=df_flow, win=win, poly=poly)
    return df_flow, df_filtered


# @st.cache_data(show_spinner='Plotting selected date...')
def select_day_plot(df_filtered, year_label):
    begin_date = df_filtered['timestamp'].iloc[0]
    end_date = df_filtered['timestamp'].iloc[-1]
    st.subheader(f'Plot for selected date in {year_label}')
    start_date = st.date_input('Start date', value=begin_date, min_value=begin_date, max_value=end_date)
    start_date = pd.Timestamp(start_date).tz_localize('Europe/Oslo')
    end_date = start_date + pd.Timedelta(days=1)

    chosen_data = df_filtered[(df_filtered['timestamp'] >= start_date) & (df_filtered['timestamp'] < end_date)]
    pos_flow_time = chosen_data[chosen_data['controller_state'] == 'OFF'][['timestamp']]
    pos_flow = chosen_data[chosen_data['controller_state'] == 'OFF'][['flow_ls']]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(chosen_data['timestamp'], chosen_data['level'], label='Level', alpha=0.9, color='gray')
    ax.plot(pos_flow_time['timestamp'], pos_flow['flow_ls'], 'o', c='orange', label='Flow (OFF)')
    ax.plot(chosen_data['timestamp'], chosen_data['sav_flow'], color='red', label='Smoothed Flow')
    ax.legend()
    st.pyplot(fig)
    return chosen_data


# @st.cache_data(show_spinner='Running simulation...')
def simulate_control_run(df_control, pump_setpoint, pump_stop, pump_data):
    sim = Control(df_control, 0, pump_setpoint, pump_stop, pump_data)
    sim.consumption_print()
    sim.plot_control()


if __name__ == '__main__':
    main()
