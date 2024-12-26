import kivy
from kivy.app import App
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.button import Button
from kivy.graphics import Line, Color, Rectangle, Ellipse, Canvas, InstructionGroup
from kivy.core.text import LabelBase
from kivy.uix.label import Label
from kivy.config import Config
from kivy.core.window import Window
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from kivy.uix.screenmanager import Screen
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import openpyxl
import numpy as np
from scipy.signal import find_peaks
import math
from kivy.uix.screenmanager import ScreenManager, Screen
from datetime import date
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
import os
from datetime import datetime
import subprocess
import sqlite3
import wfdb
from openpyxl import load_workbook
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt, decimate, welch
import scipy.signal as signal
from scipy.signal import decimate
import pywt


Window.size = (1100, 700)  # Sets the window size
Window.borderless = False   # To have borders on the window if needed
Window.resizable = False  # Prevents the window from resizing

window_length = 1
overlap = 0.5
width = 1

def ecg_filter(signal, samplerate, filter_types, lowpass_frequency=None, highpass_frequency=None, notch_frequency=None,
               filter_method='Butterworth'):
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    if signal.shape[1] > signal.shape[0]:
        signal = signal.T
        transpose_flag = True
    else:
        transpose_flag = False

    if filter_method.lower() in ['smooth', 's']:
        case_var = 1
    elif filter_method.lower() in ['gauss', 'g']:
        case_var = 2
    elif filter_method.lower() in ['butterworth', 'b']:
        case_var = 3
    else:
        raise ValueError('Filter method not recognized')

    if not np.issubdtype(signal.dtype, np.float64):
        signal = signal.astype(np.float64)

    n_samples, n_channels = signal.shape
    l = int(round(samplerate * 10))
    filteredsignal = np.pad(signal, ((l, l), (0, 0)), mode='constant')

    if lowpass_frequency and lowpass_frequency > samplerate / 2:
        lowpass_frequency = samplerate / 2 - 1
    if highpass_frequency and highpass_frequency > samplerate / 2:
        highpass_frequency = samplerate / 2 - 1

    for filter_type in filter_types:
        if filter_type == 'low':
            filteredsignal = apply_lowpass_filter(filteredsignal, samplerate, lowpass_frequency, case_var, n_channels)
        elif filter_type == 'high':
            filteredsignal = apply_highpass_filter(filteredsignal, samplerate, highpass_frequency, case_var, n_channels)
        elif filter_type == 'notch':
            filteredsignal = apply_notch_filter(filteredsignal, samplerate, notch_frequency, width)
        elif filter_type == 'band':
            if lowpass_frequency is None or highpass_frequency is None:
                raise ValueError('Both lowpass_frequency and highpass_frequency must be specified for bandpass filter.')
            filteredsignal = apply_bandpass_filter(filteredsignal, samplerate, lowpass_frequency, highpass_frequency,
                                                   case_var, n_channels)
        else:
            raise ValueError('Filter type not recognized')

    filteredsignal = filteredsignal[l:-l, :]
    filteredsignal, offset = isoline_correction(filteredsignal)

    if transpose_flag:
        filteredsignal = filteredsignal.T

    return filteredsignal

def apply_lowpass_filter(signal, samplerate, lowpass_frequency, case_var, n_channels):
    if case_var == 1:  
        nw = int(round(samplerate / lowpass_frequency))
        for i in range(n_channels):
            signal[:, i] = smooth(signal[:, i], nw)
    elif case_var == 2:  
        sigmaf = lowpass_frequency
        sigma = samplerate / (2 * np.pi * sigmaf)
        signal = gaussian_filter1d(signal, sigma, axis=0)
    elif case_var == 3:  
        order = 3
        sos = butter(order, 2 * lowpass_frequency / samplerate, btype='low', output='sos')
        for i in range(n_channels):
            signal[:, i] = sosfiltfilt(sos, signal[:, i])
           
    return signal

def apply_highpass_filter(signal, samplerate, highpass_frequency, case_var, n_channels):
    if case_var == 3:  
        order = 3
        sos = butter(order, 2 * highpass_frequency / samplerate, btype='high', output='sos')
        for i in range(n_channels):
            signal[:, i] = sosfiltfilt(sos, signal[:, i])
            
    else:
        raise NotImplementedError("High-pass filter is only implemented for Butterworth filter.")
    return signal

def apply_notch_filter(signal, samplerate, notch_frequency, width):
    K = int(np.floor(samplerate / 2 / notch_frequency))
    extpoints = int(round(0.5 * np.ceil(samplerate / width)))
    signal_extended = np.pad(signal, ((extpoints, extpoints), (0, 0)), 'symmetric')
    L = signal_extended.shape[0]  
    f = np.fft.fftfreq(L, d=1 / samplerate)  
    sigmaf = width  
    sigma = int(np.ceil(L * sigmaf / samplerate))  
    lg = 2 * round(4 * sigma) + 1  
    lb = (lg - 1) // 2  

    g = gaussian_filter1d(np.eye(1, lg).flatten(), sigma)
    g = 1 / (np.max(g) - np.min(g)) * (np.max(g) - g) 

    H = np.ones(L)  
    
    for k in range(1, K + 1):
        b = np.argmin(np.abs(f - k * notch_frequency))  
        H[b - lb:b + lb + 1] = g  
        H[L - b - lb:L - b + lb + 1] = g  

    H = np.tile(H, (signal_extended.shape[1], 1)).T  
    X = np.fft.fft(signal_extended, axis=0)  
    Y = H * X  
    signal = np.real(np.fft.ifft(Y, axis=0))  
    signal = signal[extpoints:-extpoints, :]  

    return signal

def apply_bandpass_filter(signal, samplerate, lowpass_frequency, highpass_frequency, case_var, n_channels):
    signal = apply_lowpass_filter(signal, samplerate, lowpass_frequency, case_var, n_channels)
    signal = apply_highpass_filter(signal, samplerate, highpass_frequency, case_var, n_channels)
    
    return signal

def isoline_correction(signal, number_bins=None):
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    filteredsignal = np.zeros_like(signal)
    number_channels = signal.shape[1]
   
    if number_bins is None:
        number_bins = min(2 ** 10, signal.shape[0]) 

    frequency_matrix = np.zeros((number_bins, number_channels))
    bins_matrix = np.zeros_like(frequency_matrix)
    offset = np.zeros(number_channels)

    for i in range(number_channels):
        frequency_matrix[:, i], bin_edges = np.histogram(signal[:, i], bins=number_bins)
        pos = np.argmax(frequency_matrix[:, i])  
        offset[i] = (bin_edges[pos] + bin_edges[pos + 1]) / 2  
        filteredsignal[:, i] = signal[:, i] - offset[i]  

    return filteredsignal, offset

def smooth(signal, window_len):
    s = np.r_[signal[window_len - 1:0:-1], signal, signal[-2:-window_len - 1:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len / 2 - 1):-int(window_len / 2)]

def ecg_baseline_removal(signal, samplerate, window_length, overlap):
    L = signal.shape[0]
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    NCH = signal.shape[1]

    baseline = np.zeros_like(signal)
    filtered_signal = np.zeros_like(signal)

    window_length = int(round(window_length * samplerate))
    window_length = window_length + 1 - window_length % 2
    window_half_length = (window_length - 1) // 2

    if 0 <= overlap < 1:
        N = int(np.floor((L - window_length * overlap) / (window_length * (1 - overlap))))
        center = np.round(window_length * (1 - overlap) * np.arange(N)) + window_half_length
        center = center.astype(int)
    elif overlap == 1:
        center = np.arange(1, L + 1)
        N = len(center)
    else:
        raise ValueError('overlap must be a number between 0 and 1')

    for j in range(NCH):
        baseline_points = np.zeros(center.shape)
        for i in range(N):
            leftInt = max(center[i] - window_half_length, 0)
            rightInt = min(center[i] + window_half_length, L)
            baseline_points[i] = np.median(signal[leftInt:rightInt, j])

        interpolator = PchipInterpolator(center, baseline_points)
        baseline[:, j] = interpolator(np.arange(L))
        filtered_signal[:, j] = signal[:, j] - baseline[:, j]

        corrected_signal, offset = isoline_correction(filtered_signal[:, j][:, np.newaxis])
        filtered_signal[:, j] = corrected_signal.flatten()
        baseline[:, j] += offset
        filtered_signal[:, j] += 0.05

    return filtered_signal, baseline


def butter_highpass_filter(signal, samplerate, highpass_frequency):
    order = 3
    n_channels = 1
    sos = butter(order, 2 * highpass_frequency / samplerate, btype='high', output='sos')

    if n_channels == 1:
        filtered_signal3 = sosfiltfilt(sos, signal)
    else:
        filtered_signal3 = np.zeros_like(signal)
        for i in range(n_channels):
            filtered_signal3[:, i] = sosfiltfilt(sos, signal[:, i])

    return filtered_signal3

def butter_lowpass_filter(signal, samplerate, lowpass_frequency):
    order = 3
    n_channels = 1
    sos = butter(order, 2 * lowpass_frequency / samplerate, btype='low', output='sos')

    if n_channels == 1:
        filtered_signal3 = sosfiltfilt(sos, signal)
    else:
        filtered_signal3 = np.zeros_like(signal)
        for i in range(n_channels):
            filtered_signal3[:, i] = sosfiltfilt(sos, signal[:, i])

    return filtered_signal3

def find_level(samplerate):
    x = np.ceil(np.log2(samplerate / 2 / 30))
    return int(max(x, 1))

def extend_signal(signal):
    signal_length = len(signal)
    next_pow2 = int(np.ceil(np.log2(signal_length)))

    if np.log2(signal_length) == next_pow2:
        l = 2**(next_pow2 + 1)
    else:
        l = 2**next_pow2
    
    l1 = (l - len(signal)) // 2
    l2 = l - len(signal) - l1
    ecg_w = np.pad(signal, (l1, l2), 'symmetric')
    
    return ecg_w, l1, l2

def perform_swt(ecg_w, x):
    coeffs = pywt.swt(ecg_w, 'haar', level=x)
    return coeffs

def process_signal(corrected_final_filtered_signal2, samplerate):
    
    signal = corrected_final_filtered_signal2
    signal_flattened = np.ravel(signal)
    signal_list = signal_flattened.tolist()   

    fdownsample = 400
    flagdownsample = False
    
    if samplerate > fdownsample:
        oldsamplerate = samplerate
        oldsignal = signal_list
        
        r = int(np.floor(samplerate / fdownsample))
        signal = decimate(oldsignal, r)  
        samplerate = oldsamplerate / r
        flagdownsample = True

    x = find_level(samplerate)
   
    ecg_w, l1, l2 = extend_signal(signal)
    
    coeffs = perform_swt(ecg_w, x)
    
    Dx = coeffs[-1][1]
    Dx = Dx[l2: -l1]
    
    ecg_w_flipped = np.flipud(ecg_w)
    coeffs_flipped = perform_swt(ecg_w_flipped, x)
    Dx2 = coeffs_flipped[-1][1]
    Dx2 = np.flipud(Dx2)
    Dx2 = Dx2[l2: -l1]

    Dx = np.abs(Dx + Dx2)
    std_Dx = np.std(Dx)
    
    if std_Dx > 0:
        Dx = Dx / std_Dx
        
    else:
        epsilon = 1e-10
        Dx = Dx / (std_Dx + epsilon)
        
    saturation = np.quantile(Dx, 0.99)
    Dx[Dx > saturation] = saturation

    saturation = np.max(Dx)  
    
    Thbegin = 1
    Thend = np.quantile(Dx, 0.95) / saturation
    threshold = np.linspace(Thbegin, Thend, 20)
    
    Tl = 4
    nrep = 3
    NR_vec = np.zeros(len(threshold))

    Dx = np.array(Dx) 
    n1 = int(np.floor(samplerate * Tl))
    n2 = int(np.floor(len(Dx) / n1)) - 1
    rms_Dx_base = np.zeros_like(Dx)

    if n2 == 0:
        rms_Dx_base[:] = np.quantile(Dx[int(0.1 * samplerate): len(Dx) - int(0.1 * samplerate)], 0.95)
    else:
        for i in range(n2 + 1):
            if i == 0:
                rms_Dx_base[:n1] = np.quantile(Dx[int(0.1 * samplerate): n1], 0.95)
            elif i == n2:
                rms_Dx_base[i * n1:] = np.quantile(Dx[i * n1 + 1: len(Dx) - int(0.1 * samplerate)], 0.95)
            else:
                rms_Dx_base[i * n1: (i + 1) * n1] = np.quantile(Dx[i * n1 + 1: (i + 1) * n1], 0.95)
        
    for j in range(nrep):
        for H in range(len(threshold)):
            if H == len(threshold) - 1:
                mt = np.argmin(np.diff(NR_vec[:H]))
                rms_Dx = threshold[mt] * rms_Dx_base
            else:
                rms_Dx = threshold[H] * rms_Dx_base
            
    candidates_Dx = Dx > rms_Dx
    
    Can_Sig_Dx = np.zeros_like(Dx)
    Can_Sig_Dx[candidates_Dx] = 1
    Can_Sig_Dx[0] = 0
    Can_Sig_Dx[-1] = 0

    i = np.arange(len(Can_Sig_Dx) - 1)
    Bound_A = np.where((Can_Sig_Dx[i] == 0) & (Can_Sig_Dx[i + 1] > 0))[0] + 1
    Bound_B = np.where((Can_Sig_Dx[i] > 0) & (Can_Sig_Dx[i + 1] == 0))[0]
    
    while np.min(Bound_A[1:] - Bound_B[:-1]) / samplerate < 0.1:
        ind = np.where((Bound_A[1:] - Bound_B[:-1]) / samplerate < 0.1)[0]
        Bound_B = np.delete(Bound_B, ind)
        Bound_A = np.delete(Bound_A, ind + 1)
    
    ind = np.logical_or((Bound_B - Bound_A) / samplerate < 5e-3, (Bound_B - Bound_A) / samplerate > 0.25)
    Bound_B = np.delete(Bound_B, ind)
    Bound_A = np.delete(Bound_A, ind)
    
    QRS_pos = 0.5 * (Bound_A + Bound_B)
    
    NR_vec = np.zeros(len(threshold)) 
    R_Cell = []

    for j in range(nrep):
    
        for H in range(len(threshold)):
            NR_vec[H] = len(Bound_A)

            if H > 1:
                dNR = NR_vec[H] - NR_vec[H-1]

                if dNR <= 0 or H == len(threshold) - 1:
                    if Bound_A.size == 0 or Bound_B.size == 0 or len(Bound_A) == 1 or len(Bound_B) == 1:
                        continue
                    else:
                        Tl = np.quantile(np.diff(Bound_A) / samplerate, 0.98) * 4
                        break
       
        R_Cell.append(QRS_pos)

    if flagdownsample == True:
        samplerate = oldsamplerate
        signal = oldsignal
        R_Cell_original = [QRS_pos_list * r for QRS_pos_list in R_Cell]
    else:
        R_Cell_original = R_Cell

    return R_Cell_original, samplerate, signal

def find_qrs_peaks(R_Cell_original, samplerate, signal):

    highpass_frequency = 0.5
    if samplerate > 500:
        lowpass_frequency = 250  
        filtered_signal3 = butter_highpass_filter(signal, samplerate, highpass_frequency)
        signal = butter_lowpass_filter(filtered_signal3, samplerate, lowpass_frequency)
        
    else:
        signal = butter_highpass_filter(signal, samplerate, highpass_frequency)

    num_sublists = len(R_Cell_original[0])
    
    R_Cell_column_format = []
    
    for i in range(num_sublists):
        column = [array[i] for array in R_Cell_original]
        R_Cell_column_format.append(column)
    
    R_Cell_column_format = np.array(R_Cell_column_format)
    R_Synced = R_Cell_column_format
    
    if len(R_Synced) == 0:
        print('No QRS complexes were found. Returning an empty FPT table')
        return []
    
    WB = round(0.05 * samplerate) 
    QRS_region = np.column_stack((R_Synced - WB, R_Synced + WB))
    
    if np.any(R_Synced - WB < 1):
        ind = np.where(R_Synced - WB >= 1)[0][0]
        R_Synced = R_Synced[ind:]
    
    if np.any(R_Synced[0] + WB > len(signal)):
        ind = np.where(R_Synced + WB <= len(signal))[0][-1]
        R_Synced = R_Synced[ind:]
    
    num_rows = R_Synced.shape[0] 
    FPT = np.zeros((num_rows, 13))

    if len(R_Synced) < 3:
        print('Too little QRS complexes were detected. Returning an empty FPT table')
        return []

    RPOS_vector = np.zeros(FPT.shape[0])
    QPOS_vector = np.copy(RPOS_vector)
    SPOS_vector = np.copy(RPOS_vector)    

    dsignal = np.diff(signal)
    i = np.arange(len(dsignal) - 1)
    I_ext = np.where((dsignal[i] >= 0) & (dsignal[i+1] < 0) | (dsignal[i] < 0) & (dsignal[i+1] >= 0))[0] + 1   

    R_Synced = R_Synced[:, 0].astype(int)  
    RR = np.diff(R_Synced)
    X = np.column_stack((RR[:-1], RR[1:]))
    index = np.arange(len(X))
    
    mean_X1 = np.mean(X[index, 0])
    mean_X2 = np.mean(X[index, 1])
    SCORE = (X - np.column_stack((mean_X1 * np.ones(len(X)), mean_X2 * np.ones(len(X))))) * (1 / np.sqrt(2))
    SCORE = SCORE @ np.array([[1, -1], [1, 1]])
    
    D1 = np.abs(SCORE[:, 0])
    Thl1 = 2.5 * np.std(D1)
    index = (SCORE[:, 0] >= -Thl1) & (SCORE[:, 1] <= 0)
    Ind_QRS_normal = np.where(index)[0] + 1
    Ind_QRS_normal = Ind_QRS_normal[1:-1]
    
    QRS_Matrix = np.zeros((2 * WB + 1, len(Ind_QRS_normal)))
    MP = np.zeros((len(Ind_QRS_normal), 2))
    for k in range(len(Ind_QRS_normal)):
        start_idx = R_Synced[Ind_QRS_normal[k]] - WB
        end_idx = R_Synced[Ind_QRS_normal[k]] + WB + 1
        segment = signal[start_idx:end_idx] 
        QRS_Matrix[:, k] = segment
        MP[k, :] = [np.max(segment), np.min(segment)]
    
   
    Th11 = np.quantile(MP[:, 0], 0.25)
    Th12 = np.quantile(MP[:, 0], 0.75)
    Th21 = np.quantile(MP[:, 1], 0.25)
    Th22 = np.quantile(MP[:, 1], 0.75)
    QRS_Matrix_selected = QRS_Matrix[:, (MP[:, 0] >= Th11) & (MP[:, 0] <= Th12) & (MP[:, 1] >= Th21) & (MP[:, 1] <= Th22)]
    
    if QRS_Matrix_selected.size == 0:
        Template = np.mean(QRS_Matrix, axis=1)
    else:
        Template = np.mean(QRS_Matrix_selected, axis=1)
    
    R_type = np.sign(np.max(Template) + np.min(Template))    
    biph_crit = 2 / 5  
    w_crit = 9 / 10   

    for i in range(len(RPOS_vector)):
        tmp_ZC = np.where((I_ext >= QRS_region[i, 0] - WB) & (I_ext <= QRS_region[i, 1] + WB))[0]
        
        if len(tmp_ZC) == 0:
            RPOS_vector[i] = round((QRS_region[i, 0] + QRS_region[i, 1]) / 2)
            QPOS_vector[i] = QRS_region[i, 0]  
            SPOS_vector[i] = QRS_region[i, 1]  
        elif len(tmp_ZC) == 1:  
            RPOS_vector[i] = I_ext[tmp_ZC[0]]
            QPOS_vector[i] = QRS_region[i, 0]  
            SPOS_vector[i] = QRS_region[i, 1]  
        else:
            amplitude = signal[I_ext[tmp_ZC]]
            index = np.argsort(amplitude)  
            WB = round(0.055 * samplerate) 

            if min(abs(amplitude[index[0]] / amplitude[index[-1]]), abs(amplitude[index[-1]] / amplitude[index[0]])) > biph_crit:  
                if R_type >= 0:
                    if abs(amplitude[index[-2]] / amplitude[index[-1]]) < w_crit:
                        RPOS_vector[i] = I_ext[tmp_ZC[index[-1]]]  
                        Qpeak = index[-1] - 1
                        Speak = index[-1] + 1
                    else:  
                        RPOS_vector[i] = min(I_ext[tmp_ZC[index[-1]]], I_ext[tmp_ZC[index[-2]]])  
                        Qpeak = min(index[-2], index[-1]) - 1
                        Speak = max(index[-2], index[-1]) + 1
                else:
                    if abs(amplitude[index[1]] / amplitude[index[0]]) < w_crit:
                        RPOS_vector[i] = I_ext[tmp_ZC[index[0]]] 
                        Qpeak = index[0] - 1
                        Speak = index[0] + 1
                    else:  
                        RPOS_vector[i] = min(I_ext[tmp_ZC[index[0]]], I_ext[tmp_ZC[index[1]]])  
                        Qpeak = min(index[1], index[0]) - 1
                        Speak = max(index[1], index[0]) + 1
                if Qpeak > 0:
                    QPOS_vector[i] = I_ext[tmp_ZC[Qpeak]]  
                else:
                    QPOS_vector[i] = RPOS_vector[i] - WB 
                if Speak < len(tmp_ZC):
                    SPOS_vector[i] = I_ext[tmp_ZC[Speak]]  
                else:
                    SPOS_vector[i] = RPOS_vector[i] + WB  
            elif abs(amplitude[index[-1]]) > abs(amplitude[index[0]]): 
                if abs(amplitude[index[-2]] / amplitude[index[-1]]) < w_crit:
                    RPOS_vector[i] = I_ext[tmp_ZC[index[-1]]]  
                    Qpeak = index[-1] - 1
                    Speak = index[-1] + 1
                else:  
                    RPOS_vector[i] = min(I_ext[tmp_ZC[index[-1]]], I_ext[tmp_ZC[index[-2]]])  
                    Qpeak = min(index[-2], index[-1]) - 1
                    Speak = max(index[-2], index[-1]) + 1
                if Qpeak > 0:
                    QPOS_vector[i] = I_ext[tmp_ZC[Qpeak]]  
                else:
                    QPOS_vector[i] = RPOS_vector[i] - WB  
                if Speak < len(tmp_ZC):
                    SPOS_vector[i] = I_ext[tmp_ZC[Speak]]  
                else:
                    SPOS_vector[i] = RPOS_vector[i] + WB 
            else: 
                if abs(amplitude[index[1]] / amplitude[index[0]]) < w_crit:
                    RPOS_vector[i] = I_ext[tmp_ZC[index[0]]]  
                    Qpeak = index[0] - 1
                    Speak = index[0] + 1
                else:  
                    RPOS_vector[i] = min(I_ext[tmp_ZC[index[0]]], I_ext[tmp_ZC[index[1]]])  
                    Qpeak = min(index[1], index[0]) - 1
                    Speak = max(index[1], index[0]) + 1
                if Qpeak > 0:
                    QPOS_vector[i] = I_ext[tmp_ZC[Qpeak]]  
                else:
                    QPOS_vector[i] = RPOS_vector[i] - WB 
                if Speak < len(tmp_ZC):
                    SPOS_vector[i] = I_ext[tmp_ZC[Speak]]  
                else:
                    SPOS_vector[i] = RPOS_vector[i] + WB 

    
    return QPOS_vector, SPOS_vector, RPOS_vector
    
def CalculateHRHRV(signal, QPOS_vector, SPOS_vector, RPOS_vector):
    
    time = len(signal)
    Q = np.array(QPOS_vector, dtype = int) 
    R = np.array(RPOS_vector, dtype = int)
    S = np.array(SPOS_vector, dtype = int)
    
    SizeR = len(R)
    R = [int(x) for x in R]
    Rdiff = [R[i+1] - R[i] for i in range(len(R) - 1)]
    Raverage = np.mean(Rdiff)
    Rcut = int((Raverage/2) + Raverage)
    Rout = sum(1 for d in Rdiff if d > Rcut)
    Rreal = SizeR + Rout

    bp = (Rreal * 10000) / time
    HR = round(bp * 6, 2)
    
    Remov = [x for x in Rdiff if x <= Rcut]
    HRV = [abs(Remov[i+1] - Remov[i]) for i in range(len(Remov) - 1)]
    HRV_mean = round(np.mean(Remov), 2)
    HRV_mean2 = round(np.mean(HRV), 2)
    
    HRV_std = round(np.std(HRV), 2)
    HRV_rms = round(np.sqrt(np.mean(np.square(HRV))), 2)
    
    NN20 = 0
    NN6 = 0
    
    for i in HRV:
        if i > 20:
            NN20 = NN20+1
            
        elif i > 6:
            NN6 = NN6+1
    
    HVRTotal = len(HRV)
    pNN20 = round((100 * NN20) / HVRTotal , 2)
    pNN6 = round((100 * NN6) / HVRTotal , 2)
    

    return HR, HRV, HRV_mean, HRV_std, HRV_rms, Q, R, S, Remov, NN20, pNN20, NN6, pNN6

##### Start of the tool interface ######

class MainWindow(Screen):
    def on_size(self, *args):
        pass

    def set_file_path(self):
        self.manager.current = 'parameter' 

class ParameterWindow(Screen):
    def on_size(self, *args):
        pass

    def back_path(self):
        self.manager.current = 'main'  

# ---------  Windows version ---------
    def open_file_selector(self):
        
        root = Tk()
        root.withdraw()
        
        desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

        file_path = filedialog.askopenfilename(
            initialdir=desktop_path,
            title="Select file",
            filetypes=(("Arquivos Excel", "*.xlsx;*.xls"), ("Todos os arquivos", "*.*"))
        )

        if file_path:
            self.set_file_path(file_path)

        root.destroy()

    def set_file_path(self, path):
        self.manager.get_screen('plot').set_file_path(path)
        self.manager.current = 'plot'  
#-------------------------------------------------------------

#------------- IOS version ----------------------------------
    '''def open_file_selector(self):
        layout = BoxLayout(orientation='vertical')
        
        filechooser = FileChooserListView(filters=['*.xlsx', '*.xls'])
        
        select_button = Button(text="Select")
        select_button.bind(on_release=lambda *args: self.select_file(filechooser.selection))

        layout.add_widget(filechooser)
        layout.add_widget(select_button)

        popup = Popup(title="Select file", content=layout, size_hint=(0.9, 0.9))
        popup.open()

        self.current_popup = popup

    def select_file(self, selection):
        if selection:
            file_path = selection[0]
            self.set_file_path(file_path)
        self.current_popup.dismiss() '''
#-------------------------------------------------------------

    def set_file_path(self, path):
        samplerate = self.ids.sample_rate.text
        lowpass_frequency = self.ids.low_filter.text
        highpass_frequency = self.ids.high_filter.text
        notch_frequency = self.ids.notch_filter.text
        
        plot_screen = self.manager.get_screen('plot')
        plot_screen.set_sample_rate(samplerate, lowpass_frequency, highpass_frequency, notch_frequency)
    
        self.manager.get_screen('plot').set_file_path(path)
        self.manager.current = 'plot'


class PlotWindow(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_graph = 1 
        self.data_window_start = 0  
        self.window_size = 2000  


    def back_path(self):
        self.manager.current = 'parameter' 

    def on_enter(self):
        if hasattr(self, 'signalraw'):
            self.plot_signal(self.signalraw)
    
    def on_size(self, *args):
        if hasattr(self, 'signalraw'):
            self.plot_signal(self.signalraw)

    def set_file_path(self, path):
                
        file_name = os.path.basename(path)  
        self.ids.file_path_label.text = file_name


        workbook = load_workbook(filename=path)
        sheet = workbook.active
        matriz_uma_coluna = [cell.value for cell in sheet['A'] if cell.value is not None]
        self.signalraw = np.array(matriz_uma_coluna)

        self.ids.slider.max = len(self.signalraw) - self.window_size
        self.ids.slider.value = 0

        self.apply_filters()
    
    def set_sample_rate(self, samplerate, lowpass_frequency, highpass_frequency, notch_frequency):
        
        if not samplerate:
            self.samplerate = 1000
        else:
            self.samplerate = float(samplerate)

        if not lowpass_frequency:
            self.lowpass_frequency = 100
        else:
            self.lowpass_frequency = float(lowpass_frequency)

        if not highpass_frequency:
            self.highpass_frequency = 0.5
        else:
            self.highpass_frequency = float(highpass_frequency)

        if not notch_frequency:
            self.notch_frequency = 50
        else:
            self.notch_frequency = float(notch_frequency)

    def apply_filters(self):
        if self.samplerate is None:
            samplerate = 1000
        else:
            samplerate = self.samplerate

        if self.lowpass_frequency is None:
            lowpass_frequency = 100
        else:
            lowpass_frequency = self.lowpass_frequency

        if self.highpass_frequency is None:
            highpass_frequency = 0.5
        else:
            highpass_frequency = self.highpass_frequency

        if self.notch_frequency is None:
            notch_frequency = 50
        else:
            notch_frequency = self.notch_frequency
            

        filtered_signal, baseline = ecg_baseline_removal(self.signalraw, samplerate, window_length, overlap)
        filter_types = ['low', 'high', 'notch', 'band']
        final_filtered_signal = ecg_filter(filtered_signal, samplerate, filter_types, lowpass_frequency, highpass_frequency, notch_frequency, 'Butterworth')
        final_filtered_signal2 = final_filtered_signal  

        corrected_final_filtered_signal2, offset = isoline_correction(final_filtered_signal2)

        R_Cell_original, samplerate, signal = process_signal(corrected_final_filtered_signal2, samplerate)

        QPOS_vector, SPOS_vector, RPOS_vector = find_qrs_peaks(R_Cell_original, samplerate, signal)
            
        HR, HRV, HRV_mean, HRV_std, HRV_rms, Q, R, S, Remov, NN20, pNN20, NN6, pNN6 = CalculateHRHRV(signal, QPOS_vector, SPOS_vector, RPOS_vector)
            
        
        self.corrected_final_filtered_signal2 = corrected_final_filtered_signal2
        
        self.R = R
        self.signal = signal 
        
        plot_screen = self.manager.get_screen('poincare')
        plot_screen.set_sample(Remov, HRV_mean)
              
        # Illustration of variables
        self.ids.hr_label.text = str(HR) + ' bpm'
        self.ids.average_label.text = str(HRV_mean) + ' ms'
        self.ids.sdnn_label.text = str(HRV_std) + ' ms'
        self.ids.rmsd_label.text = str(HRV_rms) + ' ms'
        self.ids.nn20_label.text = str(NN20)
        self.ids.pnn20_label.text = str(pNN20) + ' %'
        self.ids.nn6_label.text = str(NN6)
        self.ids.pnn6_label.text = str(pNN6) + ' %'

        self.plot_signal(self.signalraw)

    def on_button1_click(self):
        self.current_graph = 1
        self.ids.eixoy_num1.opacity = 1
        self.ids.eixoy_num2.opacity = 1
        self.ids.eixoy_num3.opacity = 1
        self.ids.eixoy_num4.opacity = 1
        self.plot_signal(self.signalraw)

    def plot_signal(self, signalraw):       
        
        self.ids.ArdBox.canvas.clear()
        

        with self.ids.ArdBox.canvas:
            
            start = int(self.data_window_start)
            end = int(self.data_window_start + self.window_size)
            data_to_plot = signalraw[start:end]
            time_to_plot = [start + i for i in range(len(data_to_plot))] 

            Color(1, 1, 1, 1)
            Rectangle(pos=self.ids.ArdBox.pos, size=self.ids.ArdBox.size)

            # Create the graphic
            width = self.ids.ArdBox.width
            height = self.ids.ArdBox.height
            pos_x = self.ids.ArdBox.x
            pos_y = self.ids.ArdBox.y
            
            max_data = max(data_to_plot)
            min_data = min(data_to_plot)
            
            if max_data == min_data:
                max_data = min_data + 1 

            scale_y = (height - 100) / (max_data - min_data)
            scale_x = (width - 100) / len(data_to_plot)

            Color(0, 0, 0, 1)
            Line(points=[pos_x + 50, pos_y + 50, pos_x + 50, pos_y + height - 50], width=1.5)
            Line(points=[pos_x + 50, pos_y + 50, pos_x + width - 50, pos_y + 50], width=1.5)
            for i, time in enumerate(time_to_plot):
                if time % 500 == 0:
                    x = pos_x + 50 + i * scale_x
                    if x <= pos_x + width - 50:
                        Line(points=[x, pos_y + 45, x, pos_y + 55], width=1.5)  

            num_tracos = 4  
            y_range = max_data - min_data
            espaçamento_y = y_range / (num_tracos + 1)  
            deslocamento_vertical = 30

            for j in range(1, num_tracos + 1):  
                valor_y = min_data + j * espaçamento_y
                y = pos_y + 50 + (valor_y - min_data) * scale_y + deslocamento_vertical
                if y <= pos_y + height - 50:
                    Line(points=[pos_x + 45, y, pos_x + 55, y], width=1.5) 

            Color(0, 0, 0.5, 1)
            points = []
            for i, value in enumerate(data_to_plot):
                x = pos_x + 50 + i * scale_x
                y = pos_y + 50 + (value - min_data) * scale_y
                points.extend([x, y])

            Line(points=points, width=1.5)

    def on_button2_click(self):
        self.current_graph = 2
        self.ids.eixoy_num1.opacity = 0
        self.ids.eixoy_num2.opacity = 0
        self.ids.eixoy_num3.opacity = 0
        self.ids.eixoy_num4.opacity = 0
        self.plot_signal(self.corrected_final_filtered_signal2)
 
    def plot_signal_variant(self, corrected_final_filtered_signal2):
        self.ids.ArdBox.canvas.clear()

        with self.ids.ArdBox.canvas:

            start = int(self.data_window_start)
            end = int(self.data_window_start + self.window_size)
            data_to_plot = corrected_final_filtered_signal2[start:end]
            time_to_plot = [start + i for i in range(len(data_to_plot))]  # Ajuste para o deslocamento
            
            Color(1, 1, 1, 1)
            Rectangle(pos=self.ids.ArdBox.pos, size=self.ids.ArdBox.size)

            # Create the graphic
            width = self.ids.ArdBox.width
            height = self.ids.ArdBox.height
            pos_x = self.ids.ArdBox.x
            pos_y = self.ids.ArdBox.y          
          
            max_data = max(data_to_plot)
            min_data = min(data_to_plot)
            
            if max_data == min_data:
                max_data = min_data + 1  

            scale_y = (height - 100) / (max_data - min_data)
            scale_x = (width - 100) / len(data_to_plot)

            Color(0, 0, 0, 1)
            Line(points=[pos_x + 50, pos_y + 50, pos_x + 50, pos_y + height - 50], width=1.5)
            Line(points=[pos_x + 50, pos_y + 50, pos_x + width - 50, pos_y + 50], width=1.5)

            for i, time in enumerate(time_to_plot):
                if time % 500 == 0:
                    x = pos_x + 50 + i * scale_x
                    if x <= pos_x + width - 50:
                        Line(points=[x, pos_y + 45, x, pos_y + 55], width=1.5) 
                                   
            Color(0, 0, 0.5, 1)
            points = []
            for i, value in enumerate(data_to_plot):
                x = pos_x + 50 + i * scale_x
                y = pos_y + 50 + (value - min_data) * scale_y
                points.extend([x, y])

            Line(points=points, width=1.5)

    def on_button3_click(self):
        self.current_graph = 3
        self.ids.eixoy_num1.opacity = 0
        self.ids.eixoy_num2.opacity = 0
        self.ids.eixoy_num3.opacity = 0
        self.ids.eixoy_num4.opacity = 0
        self.plot_signal_third_variant(self.signal)

    def plot_signal_third_variant(self, signal):
        self.ids.ArdBox.canvas.clear()

        with self.ids.ArdBox.canvas:
            
            start = int(self.data_window_start)
            end = int(self.data_window_start + self.window_size)
            data_to_plot = signal[start:end]
            R_peaks_in_window = [r for r in self.R if start <= r < end]
            time_to_plot = [start + i for i in range(len(data_to_plot))]  
        
            Color(1, 1, 1, 1)
            Rectangle(pos=self.ids.ArdBox.pos, size=self.ids.ArdBox.size)

            # Create the graphic
            width = self.ids.ArdBox.width
            height = self.ids.ArdBox.height
            pos_x = self.ids.ArdBox.x
            pos_y = self.ids.ArdBox.y

            max_data = max(data_to_plot)
            min_data = min(data_to_plot)
            
            if max_data == min_data:
                max_data = min_data + 1  

            scale_y = (height - 100) / (max_data - min_data)
            scale_x = (width - 100) / len(data_to_plot)

            Color(0, 0, 0, 1)
            Line(points=[pos_x + 50, pos_y + 50, pos_x + 50, pos_y + height - 50], width=1.5)
            Line(points=[pos_x + 50, pos_y + 50, pos_x + width - 50, pos_y + 50], width=1.5)

            for i, time in enumerate(time_to_plot):
                if time % 500 == 0:
                    x = pos_x + 50 + i * scale_x
                    if x <= pos_x + width - 50:
                        Line(points=[x, pos_y + 45, x, pos_y + 55], width=1.5)  
                                  
            
            Color(0, 0, 0.5, 1)
            points = []
            for i, value in enumerate(data_to_plot):
                x = pos_x + 50 + i * scale_x
                y = pos_y + 50 + (value - min_data) * scale_y
                points.extend([x, y])

            Line(points=points, width=1.5)
            
            # Mark the "R" peaks
            Color(1, 0, 0, 1)  
            for r in R_peaks_in_window:
                local_index = r - start
                x_r = pos_x + 50 + local_index * scale_x
                y_r = pos_y + 50 + (data_to_plot[local_index] - min_data) * scale_y
                Ellipse(pos=(x_r - 5, y_r - 5), size=(10, 10))  
            
    def update_graph(self, *args):
        self.data_window_start = self.ids.slider.value
        if self.current_graph == 1:
            self.plot_signal(self.signalraw)
        elif self.current_graph == 2:
            self.plot_signal_variant(self.corrected_final_filtered_signal2)
        elif self.current_graph == 3:
            self.plot_signal_third_variant(self.signal)

    def next_path(self):
        
        self.manager.current = 'poincare'
        

class PoincareWindow(Screen):
    def __init__(self, **kwargs):
        super(PoincareWindow, self).__init__(**kwargs)
        self.Remov = None
        self.HRV_mean = None

    def on_enter(self):
        self.plot_poincare_rr()

    def on_size(self, *args):
        self.plot_poincare_rr()

    def on_pos(self, *args):
        self.plot_poincare_rr()


    def set_sample(self, Remov, HRV_mean):
        
        if not Remov:
            print('erro')
        else:
            self.Remov = Remov
        
        if not HRV_mean:
            print('erro2')
        else:
            self.HRV_mean = HRV_mean


    def plot_poincare_rr(self):
        
        Remov = self.Remov
        HRV_mean = self.HRV_mean

        self.ids.ArdBox2.canvas.clear()
        self.ids.ArdBox3.canvas.clear()

        rr_n = Remov[:-1]
        rr_n1 = Remov[1:]
        
        rr_variation = Remov - HRV_mean
        rr_n_var = rr_variation[:-1]
        rr_n1_var = rr_variation[1:]

        with self.ids.ArdBox2.canvas:
            Color(1, 1, 1, 1)
            Rectangle(pos=self.ids.ArdBox2.pos, size=self.ids.ArdBox2.size)

            # Setting the position and size of ArdBox2
            width = self.ids.ArdBox2.width
            height = self.ids.ArdBox2.height
            pos_x = self.ids.ArdBox2.x
            pos_y = self.ids.ArdBox2.y

            max_rr_n_var = max(rr_n_var) +20
            min_rr_n_var = min(rr_n_var) -20
            max_rr_n1_var = max(rr_n1_var) +20
            min_rr_n1_var = min(rr_n1_var) -20

            mid_rr_n_var = (max_rr_n_var) / 2
            mid_rr_n1_var = (max_rr_n1_var) / 2

            scale_x_var = (width - 100) / (max_rr_n_var - min_rr_n_var)
            scale_y_var = (height - 100) / (max_rr_n1_var - min_rr_n1_var)

            # Create the Poincaré graph
            Color(0, 0, 1, 1)  
            for i in range(len(rr_n_var)):
                x_var = pos_x + 50 + (rr_n_var[i] - min_rr_n_var) * scale_x_var
                y_var = pos_y + 50 + (rr_n1_var[i] - min_rr_n1_var) * scale_y_var
                Ellipse(pos=(x_var - 2.5, y_var - 2.5), size=(5, 5))
            Color(0, 0, 0, 1)  
            Line(points=[pos_x + width / 2, pos_y + 50, pos_x + width / 2, pos_y + height - 50], width=1.5)
            for y in range(int(min_rr_n1_var), int(max_rr_n1_var), 5):
                y_pos = pos_y + 50 + (y - min_rr_n1_var) * scale_y_var
                Line(points=[pos_x + width / 2 - 5, y_pos, pos_x + width / 2 + 5, y_pos], width=1)  
            Color(0, 0, 0, 1)  
            Line(points=[pos_x + 50, pos_y + height / 2, pos_x + width - 50, pos_y + height / 2], width=1.5)        
            for x in range(int(min_rr_n_var), int(max_rr_n_var), 5):
                x_pos = pos_x + 50 + (x - min_rr_n_var) * scale_x_var
                Line(points=[x_pos, pos_y + height / 2 - 5, x_pos, pos_y + height / 2 + 5], width=1)             
            Color(1, 0, 0, 1)  
            Line(points=[pos_x + 50, pos_y + 50, pos_x + width - 50, pos_y + height - 50], width=1.5)
        
        # poincare graph variables
        self.ids.poinc1_label.text = f"{int(max_rr_n_var)}"
        self.ids.poinc2_label.text = f"- {int(max_rr_n_var)}"
        self.ids.poinc3_label.text = f" {int(max_rr_n1_var)}"
        self.ids.poinc4_label.text = f" - {int(max_rr_n1_var)}"
        self.ids.poinc5_label.text = f" {int(mid_rr_n_var)}"
        self.ids.poinc6_label.text = f"- {int(mid_rr_n_var)}"
        self.ids.poinc7_label.text = f" {int(mid_rr_n1_var)}"
        self.ids.poinc8_label.text = f"- {int(mid_rr_n1_var)}"


        with self.ids.ArdBox3.canvas:
            Color(1, 1, 1, 1)
            Rectangle(pos=self.ids.ArdBox3.pos, size=self.ids.ArdBox3.size)

            # Setting the position and size of ArdBox3
            width = self.ids.ArdBox3.width
            height = self.ids.ArdBox3.height
            pos_x = self.ids.ArdBox3.x
            pos_y = self.ids.ArdBox3.y

            # Set the limits for the histogram (bins)
            min_remov = min(Remov)
            max_remov = max(Remov)
            num_bins = 10  

            
            bin_width = (max_remov - min_remov) / num_bins
            bins = [min_remov + i * bin_width for i in range(num_bins + 1)]
            counts = [0] * num_bins

            for value in Remov:
                for i in range(num_bins):
                    if bins[i] <= value < bins[i + 1]:
                        counts[i] += 1
                        break
            
            # histogram variables
            self.ids.bin1_label.text = f" {counts[0]}"
            self.ids.bin2_label.text = f" {counts[1]}"
            self.ids.bin3_label.text = f" {counts[2]}"
            self.ids.bin4_label.text = f" {counts[3]}"
            self.ids.bin5_label.text = f" {counts[4]}"
            self.ids.bin6_label.text = f" {counts[5]}"
            self.ids.bin7_label.text = f" {counts[6]}"
            self.ids.bin8_label.text = f" {counts[7]}"
            self.ids.bin9_label.text = f" {counts[8]}"
            self.ids.bin10_label.text = f" {counts[9]}"

            self.ids.bin11_label.text = f"{int(bins[0])}"
            self.ids.bin12_label.text = f"{int(bins[1])}"
            self.ids.bin13_label.text = f"{int(bins[2])}"
            self.ids.bin14_label.text = f"{int(bins[3])} "
            self.ids.bin15_label.text = f"{int(bins[4])} "
            self.ids.bin16_label.text = f"{int(bins[5])} "
            self.ids.bin17_label.text = f"{int(bins[6])} "
            self.ids.bin18_label.text = f"{int(bins[7])} "
            self.ids.bin19_label.text = f"{int(bins[8])} "
            self.ids.bin20_label.text = f"{int(bins[9])} "

            # Create histogram
            max_count = max(counts)
            if max_count == 0:
                max_count = 1 
            bar_width = (width - 100) / num_bins
            scale_y = (height - 100) / max_count
            Color(0.5, 0.7, 1, 1)  
            for i in range(num_bins):
                x = pos_x + 50 + i * bar_width
                bar_height = counts[i] * scale_y
                Rectangle(pos=(x, pos_y + 50), size=(bar_width - 5, bar_height))  
            Color(0, 0, 0, 1)  
            Line(points=[pos_x + 50, pos_y + 50, pos_x + 50, pos_y + height - 50], width=1.5)
            Line(points=[pos_x + 50, pos_y + 50, pos_x + width - 50, pos_y + 50], width=1.5)

    def back_path(self):
        self.manager.current = 'plot' 


    ####### Create tool window #########
class WindowManager(ScreenManager): 
    pass

kv = Builder.load_file("ecgRatScreen.kv") #Select screen design file in kv

class BP_ECG(App):  
    def build(self):
        self.sm = ScreenManager()
        self.sm.add_widget(ParameterWindow(name='parameter'))
        self.sm.add_widget(PlotWindow(name='plot'))
        return kv

if __name__ == "__main__":
    BP_ECG().run()