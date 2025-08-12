#!/usr/bin/env python3

#---------------------------------
# Author: Ankit Anand
# Date: 10/08/25
# Email: ankit0.anand0@gmail.com
#---------------------------------


import streamlit as st
import modusa as ms
import os
import tempfile

import sys
from pathlib import Path
modules_dir = Path(__file__).resolve().parents[1]/"src"/"mfeat"
sys.path.append(str(modules_dir))

from loudness import get_loudness_contour
from high_cut_off import get_high_cutoff_freq
from tempo import get_tempo
from rhythmicity import get_rhythmicity


# Create a streamlit webapp to allow users to upload an audio file (music) and plot all the 4 features
# Loudness, High Cutoff, Tempo, Rhythmicity

# Add a title
st.title("MFeat")
st.subheader("Musical Features Extractor")

# Create upload widget with streamlit
uploaded_audio = st.file_uploader(
	label="",
	accept_multiple_files=False,
	type=["mp3", "wav"],
	help="Upload your music file in .mp3 or .wav format"
)

# Load the audio content from st uploaded file object
if uploaded_audio is not None:
	with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
		# Load the audio at 48000
		tmp.write(uploaded_audio.read())
		tmp_path = tmp.name
		
		y, sr, _ = ms.load(tmp_path, sr=48000)
		title = uploaded_audio.name
	
		st.write(f"{title}")
		st.audio(uploaded_audio)
	
		# Compute loudness
		st.subheader("Loudness Contour")
		local_loudness, local_loudness_t, _ = get_loudness_contour(tmp_path)
		fig = ms.plot1d(
			(local_loudness, local_loudness_t),
			title=title,
			ylabel="Loudness (dB)",
			xlabel="Time (sec)",
			legend="Loudness contour"
		)
		st.pyplot(fig)
		
		# Compute high cutoff
		st.subheader("High Cutoff Frequency")
		local_cutoff, local_cutoff_t, agg_cutoff, _, S = get_high_cutoff_freq(tmp_path)
		fig = ms.plot2d(
			(S[0], S[1], S[2]),
			(local_cutoff, local_cutoff_t),
			title=title,
			ylabel="Frequency (Hz)",
			xlabel="Time (sec)",
			legend="High Cutoff"
		)
		st.write(f"Overall cutoff frequency: {agg_cutoff} Hz")
		st.pyplot(fig)
		
		
		# Compute tempo
		st.subheader("Tempo")
		local_tempo, local_tempo_t, agg_tempo, confidence = get_tempo(tmp_path)
		fig = ms.plot1d(
			(local_tempo, local_tempo_t),
			title=title,
			ylabel="Tempo (BPM)",
			xlabel="Time (sec)",
			legend="Tempo"
		)
		st.write(f"Global Tempo: {agg_tempo} BPM")
		st.pyplot(fig)
		
		
		# Compute rhythmicity
		st.subheader("Rhythmicity")
		local_rhythmicity, local_rhythmicity_t, _ = get_rhythmicity(tmp_path)
		fig = ms.plot1d(
			(local_rhythmicity, local_rhythmicity_t),
			title=title,
			ylabel="Strength",
			xlabel="Time (sec)",
			legend="Rhythmicity"
		)
		st.pyplot(fig)
	
	os.remove(tmp_path)