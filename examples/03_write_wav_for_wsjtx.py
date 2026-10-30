"""Write a noisy FT8 test file that WSJT-X can open (File > Open).
Run from the repository root:  python3 examples/03_write_wav_for_wsjtx.py [SNR_dB]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from scipy.io import wavfile
import ft8, ft8dec

snr = float(sys.argv[1]) if len(sys.argv) > 1 else -15.0
x = ft8.make_slot(ft8.message_to_tones("CQ W7TLG DN13"), f0=1000.0, t0=0.5)
y = ft8dec.add_noise(x, snr, np.random.default_rng(3))
wavfile.write("test_ft8.wav", 12000, (y / abs(y).max() * 0.5 * 32767).astype(np.int16))
print(f"wrote test_ft8.wav at {snr} dB: open it in WSJT-X with File > Open")
