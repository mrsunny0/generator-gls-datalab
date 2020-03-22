import json
import os
import boto3
import io

# from layers
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks

def lambda_handler(event, context):
    
    # initialize S3 variables
    BUCKET = os.environ['BUCKET']
    DATA_PREFIX = "raw/"
    META_PREFIX = "meta/"
    
    # S3 access
    client = boto3.client('s3')
    file_name = event['file_name']
    file_data_type = ".csv"
    file_meta_type = ".json"
    file_product_type = file_name.split("_")[0].split("-")[1] # NTEST-XXX_..._
    
    # get relevant data files
    data_resp = client.get_object(Bucket=BUCKET,
        Key=DATA_PREFIX+file_name+file_data_type
    )
    # meta_resp = client.get_object(Bucket=BUCKET,
    #     Key=META_PREFIX+file_name+file_meta_type
    # )
    
    # read in data
    data_frame = pd.read_csv(data_resp['Body'])
    pressure_data = data_frame['6'].dropna()
    pressure_data = pressure_data.values
    truncate_window = 5
    pressure_data = truncate_ends(pressure_data,
                                  truncate_window,
                                  truncate_window)
    
    # set up plotting
    fig, ax = plt.subplots(1,1)
    ax.plot(pressure_data)

    # interpolate and smooth
    f = interpolate_and_oversample(pressure_data)
    N = len(pressure_data)
    over_sample_scalar = 10
    Nfold = N * over_sample_scalar
    xnew = np.linspace(0, N-1, Nfold)
    ynew = f(xnew)
    ynew = filter_data(ynew, N if N%2 == 1 else N+1)
    ax.plot(xnew, ynew)
    
    # find peaks
    peaks = return_found_peaks(ynew,
                               height=0.3,
                               width=10)
    product_max_signal = int(os.environ["{}_MAX".format(file_product_type)])
    relative_peak_scalar = 3
    base_line_noise = product_max_signal / relative_peak_scalar # use max signal for peak filtering
    peaks = reject_noisy_peaks(peaks,
                               ynew,
                               base_line_noise)
    peak_heights = ynew[peaks]
    ax.plot(xnew[peaks], peak_heights, 'ro')

    # calculate angle
    fig_ang, ax = plt.subplots(1,1)
    angle = transform_signal_to_angle(ynew,
                                      file_product_type)
    ax.plot(angle)
    
    # create statistics summary
    num_peaks = len(peaks)
    summary = {
        "number_of_peaks" : num_peaks,
        "average" : int(np.mean(angle)) if num_peaks > 0 else 0,
        "variance" : int(np.var(angle)) if num_peaks > 0 else 0,
        "std" : int(np.std(angle)) if num_peaks > 0 else 0,
        "max" : int(np.max(angle)) if num_peaks > 0 else 0,
        "min" : int(np.min(angle)) if num_peaks > 0 else 0,
        # "outliers" : -1,    
    }
    
    # create list of dictionaries
    peaks_obj = []
    for p in peaks:
        t = xnew[p]
        sig = ynew[p]
        ang = angle[p]
        peaks_obj.append({
            "time": int(np.round(t,2)),
            "signal": int(np.round(sig,2)),
            "angle": int(ang)
        })    

    # put saved data
    SAVE_DATA_PREFIX = "processed-data/"
    SAVE_META_PREFIX = "processed-meta/"
    SAVE_ASST_PREFIX = "processed-assets/"
        
    data_dump = {
        "summary" : summary,
        "peaks" : peaks_obj,
        "rawImageKey" : SAVE_ASST_PREFIX+"{}-data.png".format(file_name),
        "angleImageKey" : SAVE_ASST_PREFIX+"{}-angle.png".format(file_name)
    }
    data_dump_json = json.dumps(data_dump)
    
    
    # save figure
    data_fig = save_image(fig)
    angle_fig = save_image(fig_ang)
    
    # put assets into S3 bucket
    client.put_object(
        Bucket=BUCKET,
        Body=data_fig,
        ContentType="image/png",
        Key=SAVE_ASST_PREFIX+"{}-data.png".format(file_name)
    )
    
    client.put_object(
        Bucket=BUCKET,
        Body=angle_fig,
        ContentType="image/png",
        Key=SAVE_ASST_PREFIX+"{}-angle.png".format(file_name)
    )
    
    # save meta
    client.put_object(
        Bucket=BUCKET,
        Body=bytes(data_dump_json.encode('UTF-8')),
        Key=SAVE_META_PREFIX+"{}-meta.json".format(file_name)
    )
    
    # save angle data
    csv_buffer = io.StringIO()
    df = pd.DataFrame(angle, index=xnew)
    df.to_csv(csv_buffer)    
    client.put_object(
        Bucket=BUCKET,
        Body=csv_buffer.getvalue(),
        Key=SAVE_DATA_PREFIX+"{}-data.csv".format(file_name)
    )
    
    print(data_dump_json)
    
    return {
        "statusCode" : 200,
        "body" : data_dump_json,
    }

#%%
def save_state(client, bucket, body, name, contentType='text'): 
    client.put_object(Bucket=bucket, 
        Body=body,
        Key=name
    )

def truncate_ends(y, beg=5, end=5):
    return y[beg:(len(y) - end)]

def interpolate_and_oversample(y):
    x = range(len(y))
    f = interp1d(x, y, kind='cubic')
    return f

def filter_data(y, window=101, poly=3):
    return savgol_filter(y, window, poly)

def transform_signal_to_angle(y, product_type):
    """
    ---
    Basic interpretation
    ---
    KNEE = (signal - 5) * 135 / (20-5)
    ELBOW = (signal - 0) * 135 / (10-0)
    """
    # solve for the signal
    K = 1e3
    A = 2**10
    x_t = A * K / (A - y) - K
    
    # calculate max and min, based on empiricial data as well
    product_min = int(os.environ["{}_MIN".format(product_type)])
    product_max = int(os.environ["{}_MAX".format(product_type)])
    xmin = np.min(x_t)
    xmax = np.max(x_t)
    noise_var = np.std(x_t)
    noise_mean = np.mean(x_t)
    if xmax < product_max/2: 
        # if maximum signal is less than theoretical max signal
        xmax = product_max
    
    # construct solution matrix
    X = np.array([
        [xmin, 1],
        [xmax, 1]
    ])
    Y = np.array([0, 135])
    M = np.linalg.solve(X, Y)
    
    # solve for theta
    theta = M[0] * x_t + M[1]
    theta = theta.astype(int)
    
    return theta

def return_found_peaks(y, height=0.5, width=10, prominence=0.5):
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    peaks, _ = find_peaks(y_norm, 
                          height=height, 
                          # height=np.max(y) * 0.6,
                          # threshold=np.min(y) / 2,
                          # threshold=0.8,
                          width=width,
                           # distance=200,
                          )
    return peaks

def reject_noisy_peaks(peaks, data, threshold):
    """
    threshold is a hard-coded value based on the signal noise
    from the data
    """
    mask = data[peaks] > threshold
    return peaks[mask]

def save_image(fig):
    imgio = io.BytesIO()
    fig.savefig(imgio, format="png")
    imgio.seek(0)    
    return imgio

#%%
# TEST
if __name__ == "__main__":
    # file_name = "NTEST-KNE_1_1_strength-training_02-08-2020_1581176957359"
    # file_name = "NTEST-KNE_1_1_NONE_02-08-2020_1581171193161"
    # file_name = "NTEST-KNE_1_1_strength-training_02-08-2020_1581176856434"
    # file_name = "NTEST-KNE_1_1_strength-training_02-08-2020_1581174777001"
    # file_name = "NTEST-KNE_5_demo_physical-therapy_02-25-2020_1582663056969"
    file_name = "NTEST-KNE_6_demo_physical-therapy_02-25-2020_1582667743481"
    # file_name = "NTEST-SLV_6_demo_physical-therapy_02-25-2020_1582667648994"
    # file_name = "test"
    event = {
        "file_name" : file_name
    }
    
    # fake enviornment variables
    os.environ['BUCKET'] = 'nextiles-playground'
    os.environ['KNE_MAX'] = '20'
    os.environ['KNE_MIN'] = '5'
    os.environ['SLV_MAX'] = '300'
    os.environ['SLV_MIN'] = '100'
    
    context = []
    lambda_handler(event, context)