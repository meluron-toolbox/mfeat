#!/usr/bin/env python3

# Reference: FMP

import numpy as np
import scipy
import modusa as ms
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

def _get_tempogram_fourier(nov: np.ndarray, nov_sr: int, N: int, H: int, theta=np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
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


def _get_spectrum_based_novelty(X: np.ndarray, X_sr: int, gamma: int | None = None, M: int | None = None, norm: bool | None = None) -> (np.ndarray, int):
	"""
	Compute spectrum based novelty function.

	Parameters
	----------
	X: np.ndarray
		Spectrogram matrix.
	"""
	gamma = gamma or 10
	M = M or 40
	norm = norm or True
	
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

def _compute_autocorrelation_local(x, Fs, N, H, norm_sum=True):
	
	L_left = round(N / 2)
	L_right = L_left
	x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
	L_pad = len(x_pad)
	M = int(np.floor(L_pad - N) / H) + 1
	A = np.zeros((N, M))
	win = np.ones(N)
	if norm_sum is True:
		lag_summand_num = np.arange(N, 0, -1)
	for n in range(M):
		t_0 = n * H
		t_1 = t_0 + N
		x_local = win * x_pad[t_0:t_1]
		r_xx = np.correlate(x_local, x_local, mode='full')
		r_xx = r_xx[N-1:]
		if norm_sum is True:
			r_xx = r_xx / lag_summand_num
		A[:, n] = r_xx
	Fs_A = Fs / H
	T_coef = np.arange(A.shape[1]) / Fs_A
	F_coef_lag = np.arange(N) / Fs
	return A, T_coef, F_coef_lag

def _compute_tempogram_autocorr(nov, nov_sr, N, H, norm_sum=False, theta=np.arange(30, 601)):
	tempo_min = theta[0]
	tempo_max = theta[-1]
	lag_min = int(np.ceil(nov_sr * 60 / tempo_max))
	lag_max = int(np.ceil(nov_sr * 60 / tempo_min))
	A, T_coef, F_coef_lag = _compute_autocorrelation_local(nov, nov_sr, N, H, norm_sum=norm_sum)
	A_cut = A[lag_min:lag_max+1, :]
	F_coef_lag_cut = F_coef_lag[lag_min:lag_max+1]
	F_coef_BPM_cut = 60 / F_coef_lag_cut
	F_coef_BPM = theta
	tempogram = scipy.interpolate.interp1d(F_coef_BPM_cut, A_cut, kind='linear',
						axis=0, fill_value='extrapolate')(F_coef_BPM)
	return tempogram, T_coef, F_coef_BPM



def get_tempo(path, sr: int=48000):
	"""
	Find tempo of a song along with confidence.

	Parameters
	----------
	path: str | Path
		- Path of the audio file.
	sr: int
		- Sampling rate to load the audio in.
		- Default: 48000 Hz

	Returns
	-------
	np.ndarray
		- Local tempo
	np.ndarray
		- Time stamp
	int
		- Aggregated Tempo
	float
		- Confidence
	title
		- Title of the song
	"""
	y, sr, title = ms.load(path, sr=None)
	
	# Compute the spectrogram
	N, H = 1024, 480
	S = librosa.stft(y=y, n_fft=N, win_length=N, hop_length=H, window="hann")
	S_fr = sr / H
	
	# Compute nov
	nov, nov_sr = _get_spectrum_based_novelty(X=S, X_sr=S_fr, gamma=10, M=40, norm=True)
	nov, nov_sr = _resample_signal(y=nov, sr=nov_sr, target_sr=100)
	
	# Compute tempogram
	T, T_frames, T_tempos = _compute_tempogram_autocorr(nov=nov, nov_sr=nov_sr, N=4096, H=100, theta=np.arange(40, 600, 1))
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
			
	# Local tempo
	local_tempo = np.zeros(wrapped_T.shape[1])
	for i in range(wrapped_T.shape[1]):
		max_strength = np.max(wrapped_T[:,i]) # Max strength for a frame
		idx = np.argmax(wrapped_T[:,i])
		if max_strength < 0.4:
			local_tempo[i] = 0
		else:
			local_tempo[i] = wrapped_T_tempos[idx]
			
	# Return tempo
	final_tempo, n_repeats = scipy.stats.mode(local_tempo[local_tempo>0])
	confidence = n_repeats / len(local_tempo[local_tempo>0])
	
	return local_tempo, T_frames, int(final_tempo), confidence