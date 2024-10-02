import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj, transform, Transformer

import os
import subprocess
import json
import math
import cv2
import tkinter as tk
from tqdm import tqdm
from tkinter import StringVar, Toplevel, Label, OptionMenu, Button, filedialog


# STEP 1: Check if the necessary folders are available, if no, creat specified folder
def check_directories(initial_path,root_path,ffprobe_path):
    # Method: Check specified path whether exist
    def create_directory(path):
        if not os.path.exists(path):# If no specified path, creat folder
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

    # a)Cheack "Videos2Img" and "GPS" folder
    create_directory(os.path.join(root_path, "Videos2Img"))
    create_directory(os.path.join(root_path, "GPS"))

    # b)Check whether "CaX" folders are available
    ca_folders = [f for f in os.listdir(initial_path) if os.path.isdir(os.path.join(initial_path, f)) and f.startswith("Ca")]
    if not ca_folders:
        print("Lack of original videos data")
        return  # If no available "CaX" folder, exit

    # c)Check "ffprobe.exe", it's used to extract METAdata
    if not os.path.exists(ffprobe_path):
        print("Lack of ffprobe.exe")
        return  # If no available "ffprobe.exe", exit

    # d)Count how many "CaX" folder are, and creat correspongding number of "CaX" folder in "Videos2Img" folder
    videos2img_path = os.path.join(root_path, "Videos2Img")
    for folder in ca_folders:
        create_directory(os.path.join(videos2img_path, folder))

# STEP 2: Check original input videos and obtain begin time of each video
def check_videos_and_metadata(initial_path,ffprobe_path, ca_folders):
    errors = []
    video_metadata = {}
    timecodes = []
    FPS = []
    Video_Duration = []
    Frame_Count = []
    for folder in ca_folders:
        # Gets the full path to the current "CaX" folder (获取当前CaX文件夹的完整路径)
        folder_path = os.path.join(initial_path, folder)
        # Get all MP4 files in the folder (获取文件夹中所有的MP4文件)
        videos = [file for file in os.listdir(folder_path) if file.endswith('.MP4')]

        # count the number of video files (检查视频文件的数量)
        if len(videos) != 1:
            errors.append(f"There is error of videos in {folder} folder")
            continue  # Skip the next step and work on the next folder (跳过后续步骤，处理下一个文件夹)

        # if only one video file, ontinue processing (有且仅有一个视频文件，继续处理)
        video_path = os.path.join(folder_path, videos[0])

        # Build the ffprobe command and output it as JSON (构建 ffprobe 命令，输出格式为 JSON)
        command = [
            ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_entries", "format=tags:stream_tags",
            video_path
        ]
        """ use ffprobe to obtain the fps, total duration, total frames number """ # (使用ffprobe获取视频的帧率、总时长和总帧数)
        command2 = [
            ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,duration,nb_frames",
            "-of", "json",
            video_path
        ]
        try:
            # Get the start time of the video (获取视频开始时间)
            output = subprocess.check_output(command, text=True)
            json_data = json.loads(output)  # Convert the output to a JSON object (将输出转换为 JSON 对象)

            # Date time is tags.timecode of first 'stream' (直接访问第一个stream的tags中的timecode)
            first_stream = json_data.get('streams', [])[0]  # first stream data
            timecode = first_stream.get('tags', {}).get('timecode')
            if timecode:
                timecode_parts = timecode.split(';')[0]  # Split string with '; ' as the separator
                time_only = ':'.join(timecode_parts.split(':')[:3])  # hh,mm,ss
                timecodes.append(time_only)
                print(folder,"start time:",time_only)  # printf strat time
            else:
                print("No timecode found in the first stream.")

            # -----------------!!! obtain videos' parameters !!!-----------------
            output2 = subprocess.check_output(command2, text=True)
            video_info = json.loads(output2)
            stream = video_info['streams'][0]
            avg_frame_rate = math.ceil(eval(stream['avg_frame_rate']))  # videos' FPS
            duration = float(stream.get('duration', 0))
            nb_frames = int(stream.get('nb_frames', 0))
            FPS.append(avg_frame_rate)
            Video_Duration.append(duration)
            Frame_Count.append(nb_frames)

        except subprocess.CalledProcessError as e:
            print(f"Error running ffprobe: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    return timecodes, FPS, Video_Duration, Frame_Count, errors

# STEP 3: Set begin and end time of specified line
# a)Set time in panel
def create_time_selector():
    """ Creates a time selection window and returns the selected time """
    def submit_time():
        start_time = f"{start_hour_var.get()}:{start_minute_var.get()}:{start_second_var.get()}"
        end_time = f"{end_hour_var.get()}:{end_minute_var.get()}:{end_second_var.get()}"
        times.append((start_time, end_time))
        window.destroy()

    times = []
    window = tk.Tk()
    window.title("Select start and end time of this line")

    # Create variables to store the current selection of the options menu
    start_hour_var = StringVar(window)
    start_minute_var = StringVar(window)
    start_second_var = StringVar(window)
    end_hour_var = StringVar(window)
    end_minute_var = StringVar(window)
    end_second_var = StringVar(window)

    # Setting default values of 'begin' and 'end' time
    start_hour_var.set('12')
    start_minute_var.set('00')
    start_second_var.set('00')
    end_hour_var.set('12')
    end_minute_var.set('00')
    end_second_var.set('00')

    # Create a drop-down options menu
    OptionMenu(window, start_hour_var, *[f"{h:02}" for h in range(24)]).grid(row=1, column=4, padx=5, pady=5)
    OptionMenu(window, start_minute_var, *[f"{m:02}" for m in range(60)]).grid(row=1, column=5, padx=5, pady=5)
    OptionMenu(window, start_second_var, *[f"{s:02}" for s in range(60)]).grid(row=1, column=6, padx=5, pady=5)
    OptionMenu(window, end_hour_var, *[f"{h:02}" for h in range(24)]).grid(row=2, column=4, padx=5, pady=5)
    OptionMenu(window, end_minute_var, *[f"{m:02}" for m in range(60)]).grid(row=2, column=5, padx=5, pady=5)
    OptionMenu(window, end_second_var, *[f"{s:02}" for s in range(60)]).grid(row=2, column=6, padx=5, pady=5)

    # Window layout
    Label(window, text="Start Time").grid(row=1, column=0, columnspan=3, padx=5, pady=5)
    Label(window, text="End Time").grid(row=2, column=0, columnspan=3, padx=5, pady=5)
    Label(window, text="Hour").grid(row=0, column=4, padx=5, pady=5)
    Label(window, text="Minute").grid(row=0, column=5, padx=5, pady=5)
    Label(window, text="Second").grid(row=0, column=6, padx=5, pady=5)

    # Confirm button
    Button(window, text="Submit", command=submit_time).grid(row=3, column=3, columnspan=3, padx=10, pady=10)

    window.mainloop()
    return times
# b)Time transform
def time_to_seconds(t):
    """ Converts the time string hh:mm:ss to seconds """
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s
def seconds_to_min_sec(seconds):
    """ Convert the number of seconds to string mm:ss """
    minutes = seconds // 60  # Getting an integer minute
    remaining_seconds = seconds % 60  # Gets the number of seconds remaining
    return f"{minutes}:{remaining_seconds}"
# c)Calculate relative time in video
def calculate_video_time(start_time, end_time, video_start_time):
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
    video_start_seconds = time_to_seconds(video_start_time)

    # Calculate the 'start' and 'end' time points in the video
    start_in_video = max(0, start_seconds - video_start_seconds)
    end_in_video = max(0, end_seconds - video_start_seconds)

    return start_in_video, end_in_video

# STEP 4: Extract frames from videos according to relative begin and end time
def extract_frames(initial_path, ca_folders, start_time, end_time, fps, output_folder):
    for index, folder in enumerate(ca_folders):
        # Gets the full path to the current 'CaX' folder
        folder_path = os.path.join(initial_path, folder)
        # Get all 'MP4' files in the folder
        videos = [file for file in os.listdir(folder_path) if file.endswith('.MP4')]
        video_path = os.path.join(folder_path, videos[0])

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return

        # calculate frame numbers of 'strat' and 'end' time
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time[index] * video_fps)
        end_frame = int(end_time[index] * video_fps)
        # Calculate the number of frames that should be extracted
        total_frames = int((end_time[index] - start_time[index]) * fps)
        skip_ftames =math.ceil(video_fps/fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

        # Calculate the number of frames that should be extracted
        current_frame = start_frame
        token = start_frame
        frame_count = 0  # A token to store file number
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if current_frame >= start_frame and current_frame <= end_frame:
                    if current_frame == token:  # Skip frames according to the desired frame rate
                        height, width, _ = frame.shape
                        center_x, center_y = width // 2, height // 2
                        crop_width = int(width * 0.7 // 2)
                        crop_height = int(height * 0.7 // 2)
                        cropped_frame = frame[center_y - crop_height:center_y + crop_height,
                                              center_x - crop_width:center_x + crop_width]

                        frame_count += 1
                        frame_filename = os.path.join(output_folder[index],
                                                      f"{ca_folders[index]}_{frame_count:05d}.jpg")
                        cv2.imwrite(frame_filename, cropped_frame)
                        token = token+skip_ftames
                        pbar.update(1)  # Updating the progress bar
                else:
                    break

                current_frame += 1
                # print(current_frame)
        cap.release()
        print(f"Frames extracted to {output_folder[index]}")

# STEP 5: Process GPS data
# a)Select GPS file
def select_excel_file():
    """ A file selection window pops up to let the user select an Excel file, starting with the root directory of the project """
    root = tk.Tk()
    root.withdraw()
    # initial_dir = os.path.dirname(__file__)
    initial_dir = os.getcwd() # Gets the directory of the current script as the initial directory
    file_path = filedialog.askopenfilename(
        title="Select Excel file",
        initialdir=initial_dir,  # Setting the initial directory
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )      
    root.destroy()
    return file_path
# b)Process GPS data
def process_gps_data(file_path, line_start_time, line_end_time, gps_fps):
    """ Read 'Excel' files and process GPS data """

    df = pd.read_excel(file_path)# 加载Excel文件

    # Select relevant columns
    gps_data = df[['Y', 'X', 'Time']]
    gps_data = gps_data.rename(columns={'Y': 'LatRight', 'X': 'LonRight'})
    gps_data['Time'] = pd.to_datetime(gps_data['Time'], errors='coerce')
    first_date = gps_data['Time'].iloc[0].date()

    # Use 'datetime.time' to compare times directly
    start_time_obj = pd.to_datetime(str(first_date) + ' ' + line_start_time)
    end_time_obj = pd.to_datetime(str(first_date) + ' ' + line_end_time)
    # start_time_obj = pd.to_datetime("1900-01-01 " + line_start_time)
    # end_time_obj = pd.to_datetime("1900-01-01 " + line_end_time)

    # Process the Time column, keeping only the first data per second
    gps_data['Time'] = pd.to_datetime(gps_data['Time'], format='%H:%M:%S')
    # Filter out the data for the specified time period
    gps_line_data = gps_data[(gps_data['Time'] >= start_time_obj) & (gps_data['Time'] <= end_time_obj)]

    # Data is selected according to gps_fps
    if gps_fps == 1:            # 1fps Choose the 1st
        gps_line_data = gps_line_data.set_index('Time').resample(f'{1 / gps_fps}S').first().reset_index()
    else:
        # Sets the selection index for each second
        indices_per_second = {
            2: [0, 2],          # 2fps Choose the 1st and 3rd
            3: [0, 2, 4],       # 3fps Choose the 1st and 3rd and 5th
            4: [0, 1, 2, 3],    # 4fps Choose the 1st and 2nd and 3rd and 5th
            5: range(5)         # 5fps Choose all
        }
        indices = indices_per_second.get(gps_fps, [])

        # Select data
        selected_data = []
        for second in gps_line_data['Time'].dt.floor('S').unique():
            # Get all the data for the current second
            second_data = gps_line_data[gps_line_data['Time'].dt.floor('S') == second]
            # Check and select the data
            if len(second_data) >= max(indices, default=0) + 1:
                selected_data.extend(second_data.iloc[idx] for idx in indices if idx < len(second_data))

        gps_line_data = pd.DataFrame(selected_data) if selected_data else pd.DataFrame(columns=gps_data.columns)

    # Return processed data
    return gps_data, gps_line_data

# STEP 6：Save GPS data as CSV
# ================== (!! modify GPS parameters here, Z or Accuracy !!) ==================
def save_camera_trails_to_csv(ca_points, ca_folders, output_folder):
    """Save cameras' trajectory data to a CSV file, and the number of frames is determined by the number of columns of ca_points"""
    data = []
    # Iterate over the data for each camera
    for index, points in enumerate(ca_points):
        camera_label = ca_folders[index]  # e.g. : "Ca5", "Ca6"
        num_frames = len(points)  # The number of frames is determined according to the number of data points of the current camera
        for frame_count in range(num_frames):
            filename = f"{camera_label}_{frame_count + 1:05d}.jpg"
            x = points[frame_count, 0]
            y = points[frame_count, 1]
            # =========================== !!!!!!!!!!!! ===========================
            z = 0           # Z value is 0
            accuracy = 5    # Accuracy is 5
            # =========================== !!!!!!!!!!!! ===========================
            data.append([filename, x, y, z, accuracy])

    # 创建 DataFrame
    df = pd.DataFrame(data, columns=['Number', 'X', 'Y', 'Z', 'Accuracy'])

    # 保存到 CSV 文件
    csv_path = os.path.join(output_folder, "camera_trails.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

# main function (including calculate other cameras' trails)
def main():
    root = tk.Tk()
    root.withdraw()

    # ================== !! Number of cameras and array distance !! ==================
    Ca_Num = 4
    L = np.array([1.25, 0.75, 0.25, -0.25])  # Distance of each camera from the GPS
    Expected_fps = 2

    # ================== !! Set various of paths !! ==================
    # "I:\\2024_Japan\\2024_Japan_Rikuzentakata\\Pipeline_Test" should be replaced with your own folder
    # including 'Ca_Num' folders for all camera, and each folder including one video
    initial_path = filedialog.askdirectory(initialdir="H:\\2024_Japan\\2024_Japan_Rikuzentakata\\Pipeline_Test")

    # The following paths usually do not need to be modified
    root_path = os.getcwd()  # get current root path
    ffprobe_path = os.path.join(root_path, "ffmpeg-master-latest-win64-gpl", "bin", "ffprobe.exe")  # path of ffprobe
    ca_folders = [f for f in os.listdir(initial_path) if f.startswith('Ca')]  # path of each cameras' path


    if initial_path:
        check_directories(initial_path,root_path,ffprobe_path)
    else:
        print("No directory selected, exiting...")
    root.destroy()

    output_folders = []
    for index, ca in enumerate(ca_folders):
        output_path = os.path.join(root_path,"Videos2Img",ca_folders[index])
        output_folders.append(output_path)

    # Check the video file and get the metadata
    timecodes,FPS, Video_Time, Frame_Num,errors = check_videos_and_metadata(initial_path,ffprobe_path, ca_folders)
    if errors:
        for error in errors:
            print(error)

    timecodes[1] = timecodes[0]

    # Get the start and end time
    start_end_times = create_time_selector()
    print("Line start Time:", start_end_times[0][0])
    print("Line end Time:", start_end_times[0][1])

    StartTim_in_Videos = []
    EndTime_in_Videos = []
    for index, rate in enumerate(FPS):
        # print("Frame Rate:", FPS[index], "Duration:", Video_Time[index], "Frame Count:", Frame_Num[index])
        start_in_video, end_in_video = calculate_video_time(start_end_times[0][0], start_end_times[0][1], timecodes[index])
        StartTim_in_Videos.append(start_in_video)
        EndTime_in_Videos.append(end_in_video)
        print("Start time in video:",seconds_to_min_sec(StartTim_in_Videos[index]),"; End time in video:",seconds_to_min_sec(EndTime_in_Videos[index]))


    # process_videos(ca_folders, videos2img_path, expected_fps, start_time, end_time)
    extract_frames(initial_path, ca_folders, StartTim_in_Videos, EndTime_in_Videos, Expected_fps, output_folders)


    # Select GPS data and process it
    file_path = select_excel_file()
    if file_path:
        whole_gps_data, gps_data = process_gps_data(file_path,start_end_times[0][0], start_end_times[0][1], Expected_fps)
        # print(gps_data)
    else:
        print("No file selected")

    # ================== !! Rikuzentakada is Zone 54N !! ==================
    transformer = Transformer.from_crs("epsg:4612", "epsg:3100", always_xy=True)  # UTM zone 54N
    GPS_X, GPS_Y = transformer.transform(gps_data['LonRight'].values, gps_data['LatRight'].values)


    # ================== !! Plotting can be removed !! ==================
    plt.figure()
    plt.plot(GPS_X, GPS_Y, '-r', linewidth=1),plt.grid(True)
    plt.title('GPS original line'),plt.xlabel('X/m'),plt.ylabel('Y/m'),plt.show()
    # ================== !! Plotting can be removed !! ==================


    # Calculate trails of 4 cameras
    GPS_Trail = np.column_stack((GPS_X, GPS_Y))
    P_N = len(GPS_Trail)  # Number of trail sampling points
    Ca_Point = [np.zeros((P_N, 2)) for _ in range(Ca_Num)]

    # Calculating tangent and normal vectors for each point
    UnitTangent = np.zeros_like(GPS_Trail)
    Normal = np.zeros_like(GPS_Trail)

    for i in range(P_N - 1):
        if i < P_N - 1:
            Tangent = GPS_Trail[i + 1,:] - GPS_Trail[i,:]
        else:
            Tangent = GPS_Trail[i,:] - GPS_Trail[i - 1,:]

        UnitTangent[i,:] = Tangent / np.linalg.norm(Tangent)
        Normal[i,:] = np.array([-UnitTangent[i][1], UnitTangent[i][0]])

        if i > 1 and np.dot(Normal[i - 1,:], Normal[i,:]) < 0:
            Normal[i,:] = Normal[i - 1,:]

        if np.isnan(Normal[i]).any():
            Normal[i,:] = Normal[i - 1,:]

        for M in range(Ca_Num):
            Ca_Point[M][i,:] = GPS_Trail[i,:] + L[M] * Normal[i,:]

        if i >= 1:
            tangent_backward = GPS_Trail[i, :] - GPS_Trail[i - 1, :]
            direction_change = np.degrees(np.arctan2(tangent_backward[1], tangent_backward[0]) - np.arctan2(UnitTangent[i, 1], UnitTangent[i, 0]))
            if abs(direction_change) > 90:
                Normal[i, :] = Normal[i - 1, :]
                for M in range(Ca_Num):
                    Ca_Point[M][i, :] = GPS_Trail[i, :] + L[M] * Normal[i - 1, :]
    # Complement the endpoint coordinates of the last point
    for M in range(Ca_Num):
        Ca_Point[M][P_N - 1, :] = GPS_Trail[P_N - 1, :] + L[M] * Normal[-1, :]


    # Handling direction changes and plotting additional camera points
    plt.figure()
    plt.plot(GPS_Trail[:, 0], GPS_Trail[:, 1], '-k', linewidth=2, label='GPS Trail')
    colors = ['--b', '--r', '--g', '--y']  # Color for each camera trail

    # ================== !! Plotting can be removed !! ==================
    for M in range(Ca_Num):
        plt.plot(Ca_Point[M][:, 0], Ca_Point[M][:, 1], colors[M], linewidth=2, label=f'Camera {M+1}')
    plt.grid(True)
    plt.title('GPS and Camera Trails')
    plt.xlabel('X/m')
    plt.ylabel('Y/m')
    plt.legend()
    plt.show()
    # ================== !! Plotting can be removed !! ==================

    # ca_order = {0: "Ca8", 1: "Ca7", 2: "Ca6", 3: "Ca5"}
    ca_order = {0: "Ca5", 1: "Ca6", 2: "Ca7", 3: "Ca8"}
    output_folder = os.path.join(root_path, "GPS")
    save_camera_trails_to_csv(Ca_Point, ca_order, output_folder)


# Run main
main()
