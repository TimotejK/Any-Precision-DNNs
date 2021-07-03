import math
import threading
import time
from torchvision.utils import save_image
from datetime import datetime

import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import torch
from PIL import Image

import filterRawData as frd
import featureGeneration as fg

############################## Constants #############################
import models
from optimization_selection.lda_hierarchical_selector2 import LDAAccuracySelector
from NeoSensors import Accel, Gyro

path="exp01_user01.txt" # test data file

optimization_selector = LDAAccuracySelector(use_features=True)
sampling_freq=50 # 50 Hz(hertz) is sampling frequency: the number of captured values of each axial signal per second.
freq1 = 0.3 # freq1=0.3 hertz [Hz] the cuttoff frequency between the DC compoenents [0,0.3] and the body components[0.3,20]hz
freq2 = 20  # freq2= 20 Hz the cuttoff frequcency between the body components[0.3,20] hz and the high frequency noise components [20,25] hz

#Size of chunk to remove noise and gravity
data_chunk_size=449

# Window size per original dataset
w_s=128

# Overlap per original dataset
overlap = 64

bit_width_list = [0.35,0.5,0.75,1.0]
class_number = 6

model = models.__dict__["slimmableMobileNetV2"](bit_width_list, class_number)
checkpoint = torch.load("results/mobilenet_slimmable/ckpt/model_latest.pth.tar", map_location=torch.device('cpu'))
# print(checkpoint)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()

labels=["Walking", "Walking upstairs", "Walking downstairs", "Sitting", "Standing", "Laying"]

#data column names
raw_acc_columns=['acc_X','acc_Y','acc_Z']
raw_gyro_columns=['gyro_X','gyro_Y','gyro_Z']
column_names= raw_acc_columns + raw_gyro_columns

mean = [x / 255 for x in [127, 127, 127]]
std = [x / 255 for x in [45, 45, 45]]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

acceleration_conversion = 16384.0
ang_velocity_conversion = 2**16 / (2 * 250 * math.pi / 180) / 1
acc = Accel()
gyro = Gyro()

optimization_selector.init(bit_width_list)

column_names_old=['t_body_acc_X','t_grav_acc_X', 't_total_acc_X','t_body_acc_jerk_X',
                  't_body_acc_Y','t_grav_acc_Y', 't_total_acc_Y','t_body_acc_jerk_Y',
                  't_body_acc_Z','t_grav_acc_Z', 't_total_acc_Z','t_body_acc_jerk_Z',
                  't_body_gyro_X','t_body_gyro_jerk_X','t_body_gyro_Y',
                  't_body_gyro_jerk_Y','t_body_gyro_Z','t_body_gyro_jerk_Z']

column_names_new=['t_body_acc_X','t_body_acc_Y','t_body_acc_Z',
                  't_grav_acc_X','t_grav_acc_Y','t_grav_acc_Z',
                  't_body_acc_jerk_X','t_body_acc_jerk_Y','t_body_acc_jerk_Z',
                  't_body_gyro_X','t_body_gyro_Y','t_body_gyro_Z',
                  't_body_gyro_jerk_X','t_body_gyro_jerk_Y','t_body_gyro_jerk_Z']

thread_running = False

def clamp(n, minn, maxn):
    return max(min(maxn, int(n)), minn)

def reshape_row(row):
    # scale
    std = 0.3
    row = np.array(list(map(lambda x: clamp(128 + (x / (2 * std)) * 128, 0, 255), row)))
    size = 32
    if len(row) > size ** 2:
        row = row[:size ** 2]
    number_of_rows = math.ceil(len(row) / size)
    padded_row = np.zeros(size * number_of_rows)
    padded_row[:len(row)] = row
    rows = padded_row.reshape(number_of_rows, size)
    result = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        j = int(number_of_rows * i / size)
        result[i, :] = rows[j]
    return result

def signal_to_picture(signal):
    rows = len(signal)
    frames = [[]]
    for cursor in range(0, rows - 127, 64):
        if len(frames[-1]) > 380:
            frames.append([])
        # end_point: cursor(the first index in the window) + 128
        end_point = cursor + 128  # window end row

        # selecting window data points convert them to numpy array to delete rows index
        data = signal[cursor:end_point]
        frames[-1] = np.concatenate((frames[-1], data))
    return frames

def inference(data_chunk):
    global thread_running
    thread_running = True
    time_sig = np.empty((data_chunk_size - 1, len(column_names_old)))
    c_i = 0
    for col in range(np.shape(data_chunk)[2]):
        column = data_chunk[:, 0, col]
        # perform median filtering
        med_filter_col = frd.median(column)
        if 'acc' in column_names[col]:
            # perform component selection
            total_acc, grav_acc, body_acc, _ = frd.components_selection_one_signal(med_filter_col, sampling_freq, freq1,
                                                                                   freq2)

            # compute jerked signal
            body_acc_jerk = frd.jerk_one_signal(body_acc,
                                                sampling_freq)  # apply the jerking function to body components only

            # store signal in time_sig and delete the last value of each column
            # jerked signal will have the original lenght-1(due to jerking)
            time_sig[:, c_i] = body_acc[:-1]
            c_i += 1

            time_sig[:, c_i] = grav_acc[:-1]
            c_i += 1

            time_sig[:, c_i] = total_acc[:-1]
            c_i += 1

            # store body_acc_jerk signal
            time_sig[:, c_i] = body_acc_jerk
            c_i += 1

        elif 'gyro' in column_names[col]:

            # perform component selection
            _, _, body_gyro, _ = frd.components_selection_one_signal(med_filter_col, sampling_freq, freq1,
                                                                     freq2)

            # compute jerk signal
            body_gyro_jerk = frd.jerk_one_signal(body_gyro, sampling_freq)

            # store gyro signal
            time_sig[:, c_i] = body_gyro[:-1]
            c_i += 1

            # store body_gyro_jerk
            time_sig[:, c_i] = body_gyro_jerk
            c_i += 1

    # create new dataframe to order columns
    time_sig_df = pd.DataFrame()
    for col in column_names_new:  # iterate over each column in the new order
        time_sig_df[col] = time_sig[:, column_names_old.index(col)]  # store the column in the ordred dataframe

    # generate magnitude signals
    for i in range(0, 15, 3):  # iterating over each 3-axial signals

        mag_col_name = column_names_new[i][
                       :-1] + 'mag'  # create the magnitude column name related to each 3-axial signals

        col0 = np.array(time_sig_df[column_names_new[i]])  # copy X_component
        col1 = time_sig_df[column_names_new[i + 1]]  # copy Y_component
        col2 = time_sig_df[column_names_new[i + 2]]  # copy Z_component

        mag_signal = frd.mag_3_signals(col0, col1, col2)  # calculate magnitude of each signal[X,Y,Z]
        time_sig_df[mag_col_name] = mag_signal  # store the signal_mag with its appropriate column name

    # apply sliding window
    body_acc_X = signal_to_picture(time_sig[:, column_names_old.index("t_body_acc_X")])
    body_acc_Y = signal_to_picture(time_sig[:, column_names_old.index("t_body_acc_Y")])
    body_acc_Z = signal_to_picture(time_sig[:, column_names_old.index("t_body_acc_Z")])
    body_gyro_X = signal_to_picture(time_sig[:, column_names_old.index("t_body_gyro_X")])
    body_gyro_Y = signal_to_picture(time_sig[:, column_names_old.index("t_body_gyro_Y")])
    body_gyro_Z = signal_to_picture(time_sig[:, column_names_old.index("t_body_gyro_Z")])
    total_acc_X = signal_to_picture(time_sig[:, column_names_old.index("t_total_acc_X")])
    total_acc_Y = signal_to_picture(time_sig[:, column_names_old.index("t_total_acc_Y")])
    total_acc_Z = signal_to_picture(time_sig[:, column_names_old.index("t_total_acc_Z")])

    image1 = np.array([[reshape_row(np.concatenate((body_acc_X[0], body_gyro_X[0], total_acc_X[0]))),
                        reshape_row(np.concatenate((body_acc_Y[0], body_gyro_Y[0], total_acc_Y[0]))),
                        reshape_row(np.concatenate((body_acc_Z[0], body_gyro_Z[0], total_acc_Z[0])))]])
    image1 = Image.fromarray(image1.transpose((0, 2, 3, 1))[0])
    image1 = transform(image1).unsqueeze(0)
    image2 = np.array([[reshape_row(np.concatenate((body_acc_X[1], body_gyro_X[1], total_acc_X[1]))),
                        reshape_row(np.concatenate((body_acc_Y[1], body_gyro_Y[1], total_acc_Y[1]))),
                        reshape_row(np.concatenate((body_acc_Z[1], body_gyro_Z[1], total_acc_Z[1])))]])
    image2 = Image.fromarray(image2.transpose((0, 2, 3, 1))[0])
    image2 = transform(image2).unsqueeze(0)

    now = datetime.now()
    save_image(image1, 'images/' + now.strftime("%H:%M:%S") + '-1.png')
    save_image(image2, 'images/' + now.strftime("%H:%M:%S") + '-2.png')

    t_W_dic = frd.Windowing(time_sig_df)

    # conctenate all features names lists
    all_columns = fg.time_features_names()
    t_W_dic.pop("t_W00000")
    t_W_dic.pop("t_W00002")
    t_W_dic.pop("t_W00003")
    t_W_dic.pop("t_W00005")
    # apply datasets generation pipeline to time and frequency windows
    Dataset = fg.Dataset_Generation_PipeLine(t_W_dic)
    Dataset = Dataset.to_numpy()
    # network_width = optimization_selector.select_optimization_level(image1, Dataset)
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    print(labels[int(torch.argmax(model(image1)[0]))])
    print(labels[int(torch.argmax(model(image2)[0]))])
    # print("Thread finished")
    thread_running = False


def main():
    # init vars
    raw_data = []
    samples_total = 0
    samples_chunk = 0
    blocks = 0
    # open the txt file with samples
    # file = open(path, 'r')
    while True:
        sample = []
        sample.append([x / acceleration_conversion for x in acc.get()] + [x / ang_velocity_conversion for x in gyro.get()])
        # print(sample)
        raw_data.append(sample)
        samples_total = samples_total + 1
        samples_chunk = samples_chunk + 1
        # when chunk of data is full perform denoising, generate windowed data and features
        if samples_chunk == data_chunk_size:
            if samples_total > data_chunk_size:
                samples_chunk = overlap
                data_chunk = np.array(raw_data[samples_total - data_chunk_size - overlap:samples_total - overlap])
            else:
                samples_chunk = 0
                data_chunk = np.array(raw_data)
            if thread_running:
                print("Inference too slow skipping sample")
            else:
                # print("Starting thread")
                th = threading.Thread(target=inference, args=(data_chunk,))
                th.start()
        time.sleep(0.02)

if __name__ == "__main__":
    main()