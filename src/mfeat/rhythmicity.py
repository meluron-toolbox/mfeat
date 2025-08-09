#!/usr/bin/env python3

# Reference: FMP

import numpy as np
import modusa as ms
import scipy
import librosa

def _resample_signal(y, sr, target_sr):
	"""
	Resample signal to a target sampling rate.
	
	Parameters
	----------
	y: np.ndarray
		- Signal to resample.
	sr: float
		- Sampling rate of the signal.
	target_sr: float
		- Target sampling rate.

	Returns
	-------
	np.ndarray
		Resampled signal.
	float:
		New sampling rate.
	"""
	t_inp = np.arange(len(y)) * (1 / sr)
	dur_inp_sec = len(y) * (1 / sr)  # Duration of the signal
	
	N_out = int(np.ceil(dur_inp_sec * target_sr))
	t_out = np.arange(N_out) * (1 / target_sr)
	
	if t_out[-1] > t_inp[-1]:
		t_inp = np.append(t_inp, t_out[-1])
		y = np.append(y, y[-1])  # use y[-1] or 0 depending on context
		
	interpolator = scipy.interpolate.interp1d(t_inp, y, kind="linear")
	y_resampled = interpolator(t_out)
	
	return y_resampled, target_sr

def _compute_tempogram_fourier(nov: np.ndarray, nov_sr: int, N: int, H: int, theta=np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
	"""
	Get tempogram using fourier method on novelty function.

	Parameters
	----------
	nov: np.ndarray
		- Novelty function.
	nov_sr: int
		- Feature sampling rate for novelty.
	N: int
		- Window size in samples.
	H: int
		- Hop size in samples.
	theta: np.ndarray
		- What tempi to consider.
	
	Returns
	-------
	np.ndarray
		- Tempogram matrix.
	np.ndarray
		- Time axis array (sec).
	np.ndarray
		- Tempo axis array (BPM).
	"""
	win = np.hanning(N)
	N_left = N // 2
	L = nov.shape[0]
	L_left = N_left
	L_right = N_left
	L_pad = L + L_left + L_right
	x_pad = np.concatenate((np.zeros(L_left), nov, np.zeros(L_right)))
	t_pad = np.arange(L_pad)
	M = int(np.floor(L_pad - N) / H) + 1
	K = len(theta)
	X = np.zeros((K, M), dtype=np.complex128)
	
	for k in range(K):
		omega = (theta[k] / 60) / nov_sr
		exponential = np.exp(-2 * np.pi * 1j * omega * t_pad)
		x_exp = x_pad * exponential
		for n in range(M):
			t_0 = n * H
			t_1 = t_0 + N
			X[k, n] = np.sum(win * x_exp[t_0:t_1])
		T_coef = np.arange(M) * H / nov_sr
		F_coef_BPM = theta
		
	return X, T_coef, F_coef_BPM


def _get_spectrum_based_novelty(X, X_sr, gamma = 10, M = 40, norm = True) -> (np.ndarray, int):
	"""
	Compute spectrum based novelty function.

	Parameters
	----------
	X: np.ndarray
		Spectrogram matrix.
	X_sr: int
		Spectrograme frame rate.
	gamma: float
		- Log compression factor.
	M: int
		- Local averaging
	norm: bool
		- Normalise the novelty spectrum.

	Returns
	-------
	np.ndarray
		- Novelty function.
	int
		- Novelty frame rate.
	"""
	def compute_local_average(x, M):
		L = len(x)
		local_average = np.zeros(L)
		for m in range(L):
			a = max(m - M, 0)
			b = min(m + M + 1, L)
			local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
		return local_average
	
	Y = np.log(1 + gamma * np.abs(X))
	Y_diff = np.diff(Y)
	Y_diff[Y_diff < 0] = 0
	novelty_spectrum = np.sum(Y_diff, axis=0)
	novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
	if M > 0:
		local_average = compute_local_average(novelty_spectrum, M)
		novelty_spectrum = novelty_spectrum - local_average
		novelty_spectrum[novelty_spectrum < 0] = 0.0
	if norm:
		max_value = max(novelty_spectrum)
		if max_value > 0:
			novelty_spectrum = novelty_spectrum / max_value
	return novelty_spectrum, X_sr

def get_rhythmicity(path, sr=48000):
	"""
	Compute local rhythmicity of a song.

	Parameters
	----------
	path: str
		- Path of the song file.
	sr: int
		- Sampling rate to load the audio signal.
		- Default: 48000 Hz

	Returns
	-------
	np.ndarray
		- Local rhytmicity
	np.ndarray
		- Time stamp
	str
		- Title of the song
	"""
	# Load the signal
	y, sr, title = ms.load(path, sr=None)
	
	# Compute the spectrogram
	N, H = 1024, 480
	S = librosa.stft(y=y, n_fft=N, win_length=N, hop_length=H, window="hann")
	S_fr = sr / H
	
	# Compute nov
	nov, nov_fr = _get_spectrum_based_novelty(X=S, X_sr=S_fr, gamma=10, M=40, norm=True)
	nov, nov_fr = _resample_signal(y=nov, sr=nov_fr, target_sr=100)
	
	# Compute tempogram
	N, H = 1024, 100 # 10 secs, 1 sec
	T, T_frames, T_tempos = _compute_tempogram_fourier(nov=nov, nov_sr=nov_fr, N=N, H=H, theta=np.arange(40, 600, 1))
	T_fr = nov_fr / H
	T = np.abs(T) # Magnitude
	T = T / np.max(T) # Normalisation
	threshold = 0.2
	T[T < 0.2] = 0
	
	# Wrap octave
	tempo_min, tempo_max = 80, 160
	wrapped_T_tempos = np.arange(tempo_min, tempo_max)
	wrapped_T = np.zeros((tempo_max - tempo_min, T.shape[1]), dtype=T.dtype)
	
	for i, tempo in enumerate(T_tempos):
		folded_tempo = tempo
		while folded_tempo >= tempo_max:
			folded_tempo /= 2
		while folded_tempo < tempo_min:
			folded_tempo *= 2
			
		if tempo_min <= folded_tempo < tempo_max:
			idx = np.argmin(np.abs(wrapped_T_tempos - folded_tempo))
			wrapped_T[idx, :] += T[i, :]
			
	# Aggregate the strengths
	percussive_activity = np.zeros(wrapped_T.shape[1])
	for frame in range(wrapped_T.shape[1]):
		percussive_activity[frame] = np.sum(wrapped_T[:,frame])
	percussive_activity /= np.max(percussive_activity)
	
	# Compute energy
	energy = np.sum(np.abs(S)**2, axis=0)
	energy = energy / np.max(energy)
	energy_fr = S_fr
	
	win_size = 100 # 1 sec
	dynamic_activity = np.zeros_like(energy)
	for i in range(energy.shape[0]):
		start_sample = i
		end_sample = start_sample + win_size
		windowed_signal = energy[start_sample:end_sample]
		max_value = np.max(windowed_signal)
		min_value = np.min(windowed_signal)
		dynamic_activity[i] = (max_value - min_value)
		
	dynamic_activity, dynamic_activity_fr = _resample_signal(dynamic_activity, energy_fr, 1)
	
	local_rhytmicity = percussive_activity * dynamic_activity
	local_rhytmicity = scipy.signal.medfilt(local_rhytmicity, kernel_size=5) # 5 sec
	local_rhytmicity = local_rhytmicity / np.max(local_rhytmicity)
	local_rhytmicity_fr = dynamic_activity_fr
	
	t = np.arange(local_rhytmicity.size) / local_rhytmicity_fr
	
	return local_rhytmicity, t, title