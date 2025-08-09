#!/usr/bin/env python3

import numpy as np
import modusa as ms
import pyloudnorm as pyln

def _get_res_setting(res:str=None):
	"""
	Get predefined resolution settings.
    
	Parameters
	----------
	res: str
		LOW or HIGH resolution
    
	Returns
	-------
    dict
        dictionary will all values for that resolution
	"""
	if res == "HIGH":
		return dict(
            name="highres", 
            blocksize=0.4, # standard block size as per BS.1770
            winsize_sec=1., # 1 s window-size
            hopsize_sec=1., # no overlap
        )
    
	elif res == "LOW":
		return dict(
            name="highres", 
            blocksize=0.4, # standard block size as per BS.1770
            winsize_sec=5., # 1 s window-size
            hopsize_sec=5., # no overlap
        )


def get_loudness_contour(path, sr=48000, resolution="HIGH"):
    """
    Compute loudness contour of an audio signal.

    Paramters
    ---------
    path: str
        - Filepath of the audio.
    sr: int
        - Sampling rate of the audio.
        - Default: 48000 => Load the audio at 48000.
    resolution: str
        - Resolution of the loudness contour. ("LOW" or "HIGH")
        - Default: "HIGH"
        - "LOW" => 5 sec; "HIGH" => 1 sec
    Returns
    -------
    np.ndarray
        - Loudness contour.
    np.ndarray
        - Time stamp (sec).
    str
        - File name of the audio.
    """
    
    # Load the audio
    y, sr, title = ms.load(path, sr=sr)
    
    # Load the resolution settings
    res_settings = _get_res_setting(resolution)
    
    block_size = res_settings['blocksize']
    winsize_sample = int(res_settings["winsize_sec"]) * sr
    hopsize_sample = int(res_settings["hopsize_sec"]) * sr
    
    # Creating loudness meter object
    loudness_meter = pyln.Meter(sr, block_size=block_size) # BS.1770 loudness meter object (400 ms block size)
    
    # Computing it locally
    len_pad = int(np.round(winsize_sample / 2))
    y_padded = np.concatenate((np.zeros(len_pad), y, np.zeros(len_pad)))
    
    n_hops = int(np.floor((y_padded.size - len_pad) / hopsize_sample))
    loudness_contour = np.zeros(n_hops + 1)
    
    for i in range(n_hops + 1): # +1 is due to python zero indexing, if n_hops is 1, it means we get two windows (0 and 1)
        start_idx = i * hopsize_sample
        end_idx = start_idx + winsize_sample
    
        windowed_y = y_padded[start_idx:end_idx]
        local_loudness = loudness_meter.integrated_loudness(windowed_y)
        loudness_contour[i] = local_loudness
    
    loudness_contour = np.clip(loudness_contour, a_min=-70, a_max=None) # clip silence regions to -70
    loudness_countour_sr = sr / hopsize_sample
    
    loudness_countour_t = np.arange(loudness_contour.size) * (hopsize_sample / sr)
    
    return loudness_contour, loudness_countour_t, title