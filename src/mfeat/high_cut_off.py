#!/usr/bin/env python3

import numpy as np
import modusa as ms
import scipy
import librosa

# Utility
def apply_adaptation_filter(y:np.ndarray, sr:int, filter_len:int):
	"""
	Compute smoothened derivative of the signal using two gaussian profiles.

	Parameters
	----------
	y: np.ndarray
		Signal to find derivative of.
	sr: int
		Sampling rate of the signal.
	filter_len: int
		Length of the filter in samples.
	
	Returns
	-------
	np.ndarray
		Derivative of the signal.
	np.ndarray
		Kernel used to compute the derivative, one can plot the kernel to better understand the process or perform parameter tuning appropriately.
	"""
	tau1 = int(3 * sr) # Standard deviation
	d1 = int(1 * sr) # Distance from the center
	tau2 = int(3 * sr) # Standard deviation
	d2 = int(1 * sr) # Distance from the center
	
	kernel = np.zeros(2 * filter_len)
	t = np.arange(-filter_len, +filter_len+1) 
	kernel = (1/(tau1*np.sqrt(2*np.pi))) * np.exp(-(t-d1)**2/(2*tau1**2)) - (1/(tau2*np.sqrt(2*np.pi))) * np.exp(-(t+d2)**2/(2*tau2**2))
	kernel =  np.exp(-(t-d1)**2/(2*tau1**2)) - np.exp(-(t+d2)**2/(2*tau2**2))
	kernel /= np.sum(np.abs(kernel)) # Normalise the kernal
	
	# Apply the biphasic filter using convolution
	der = scipy.signal.convolve(y, kernel[::-1], mode='same') # Reversed to perform convolution in the right orientation
	der[der>0] = 0 # For this task we are only interested in drop in intensity
	
	return der, kernel


def get_high_cutoff_freq(path, sr=None):
	"""
	Find the high cutoff frequency for a given audio signal.

	Parameters
	----------
	path: str
		- Path to the audio file.
	sr: int
		- Sampling rate to load the audio in.
		- Default: None => Load with the original sampling rate.

	Returns
	-------
	np.ndarray
		- Frame level cutoff frequency (Hz)
	np.ndarray
		- Timestamp for each frame (sec)
	float:
		- Aggregated cutoff value for the audio file.
	"""
	# Load the audio signal
	y, sr, title = ms.load(path, sr=None) # Loads in mono
	
	# Compute spectrogram
	N = int(0.5 * sr) # choosing fairly long window size 
	H = int(0.5 * sr)# no overlap
	Nfft = int(2**np.ceil(np.log2(N))) # power of 2 for efficient FFT computation
	
	eps = np.finfo(float).eps # 2.22e-16    
	spec = librosa.stft(y=y, n_fft=Nfft, win_length=N, window="hann", hop_length=H)
	mag_spec = np.abs(spec)
	spec_fr = H / sr
	f = np.arange(mag_spec.shape[0]) * (sr / Nfft)
	t = np.arange(mag_spec.shape[1]) * (H / sr)
	
	downsample = 4 # since we are taking longer window, we downsample it by a factor of 4
	mag_spec = mag_spec[::downsample,:] # lower the resolution of frequency
	f = f[::4]
	freq_sr = (sr / Nfft) * downsample
	
	power_spec = mag_spec**2
	power_spec = power_spec / np.max(power_spec) # normalise [0 to 1]
	power_spec = np.clip(power_spec, a_min=eps, a_max=None) # making sure that the range is [eps, 1]
	power_spec_db = 10*np.log10(power_spec) # get values in dB [-150, 0]
	
	ders = np.zeros_like(power_spec_db)
	cutoff = np.zeros(power_spec_db.shape[1])
	
	for frame in range(power_spec_db.shape[1]): # finding the smoothened derivative along frequency dimension, we go through each time frame
		der, _ = apply_adaptation_filter(y=power_spec_db[:,frame], sr=freq_sr, filter_len=int(10*freq_sr)) # find smoothened derivative along freq axis
		ders[:, frame] = der
	
	for frame in range(power_spec_db.shape[1]): # go through each time frame
		# peak peaking
		peaks, props = scipy.signal.find_peaks(-ders[:,frame], prominence=0.2)
		proms = props["prominences"]
		if len(peaks) != 0: # Peaks are found for that frame
			max_proms_idx = np.argmax(proms) # find the freq index where there is maximum drop
	
			max_peak_idx = peaks[max_proms_idx]
			cutoff[frame] = f[max_peak_idx]
	cutoff = np.round(cutoff / 100) * 100

	cutoff = cutoff[::2] # Downsample
	cutoff_t = np.arange(cutoff.size) * (spec_fr) * 2
	# Apply median filter to remove sudden value jumps
	cutoff = scipy.signal.medfilt(cutoff, kernel_size=3)
	audio_cutoff = scipy.stats.mode(cutoff)[0]
	
	return cutoff, cutoff_t, audio_cutoff, title, [power_spec_db, f, t]