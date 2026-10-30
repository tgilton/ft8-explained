# Getting started

This guide is for readers of the white paper ([PDF](How_FT8_Really_Works.pdf)) who want to run the code, poke at it, and
reproduce the paper's figures. You need basic Python; no radio, no WSJT-X, and no signal-processing background.

## 1. What you need

- Python 3 with `numpy`, `scipy` and `matplotlib`. The code was developed and tested with Python 3.13, numpy 2.5,
  scipy 1.18 and matplotlib 3.11; other recent versions should work.
- About 100 MB of disk space and a few minutes. Nothing here transmits or touches a radio.
- Optional: [WSJT-X](https://wsjt.sourceforge.io/) (to open the test audio file in Example 3) and a C compiler plus
  [`ft8_lib`](https://github.com/kgoba/ft8_lib) (only to repeat the encoder validation in section 6).

## 2. Install

```bash
git clone https://github.com/tgilton/ft8-explained
cd ft8-explained
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run everything from the repository root (the folder that contains `ft8.py`). The modules import each other and read
`ldpc_tables.npz` from the same folder.

Check that it works:

```bash
python3 ft8.py "CQ W7TLG DN13"     # prints the 79 tones of the transmission
python3 ft8dec.py                  # self-test: decodes the message at 0, -10, -15, -18, -20 and -22 dB
```

The first command should print `3140652000000001004024452502314335263140652542352241012370734366635211003140652`.
The self-test should decode down to -18 dB and fail at -20 and -22 dB, which is the cliff described in Section 9.5 of the paper.

## 3. Where the paper's ideas live in the code

| Paper section | What it describes | Code |
|---|---|---|
| 2. The signal | 8 tones, 6.25 Hz spacing, Gaussian smoothing | `ft8.synth()`, `ft8.gfsk_pulse()` |
| 4. The 15 s cycle | transmission starts 0.5 s into a 15 s slot | `ft8.make_slot()` |
| 5. 77 bits | callsign, grid and report packing | `ft8.pack_message()`, `ft8.pack28()`, `ft8.packgrid()`, `ft8dec.unpack77()` |
| 6. CRC | 14-bit checksum | `ft8.crc14()` |
| 7. LDPC | 91 -> 174 bits; belief propagation | `ft8.ldpc_encode()`, `ft8dec.bp_decode()` |
| 8. Costas array | sync pattern; time and frequency search | `ft8.COSTAS`, `ft8dec.coarse_sync_fast()`, `ft8dec.fine_sync()` |
| 3, 7. What the decoder sees | tone energies and soft bit confidences | `ft8dec.tone_matrix()`, `ft8dec.symbol_llrs()` |
| 9. SNR | noise at a chosen SNR in 2500 Hz | `ft8dec.add_noise()`, `sim_snr.py`, `capacity.py` |

## 4. Three examples to run first

Each is a short script in `examples/`. Run them from the repository root.

**Example 1: how weak a signal can the receiver still decode?**

```bash
python3 examples/01_encode_noise_decode.py
```

Typical output:

```
SNR (dB in 2500 Hz)   decoded?
       -16            True   CQ W7TLG DN13
       -18            True   CQ W7TLG DN13
       -19            True   CQ W7TLG DN13
       -20            False  -
       -21            False  -
```

Change the seed in `np.random.default_rng(1)` and the SNR list, and watch the edge move by a dB or so.
Try your own message in `MESSAGE`; a standard call-and-grid message such as `K1ABC W7TLG R-07`, or up to 13 characters of free text, works.

**Example 2: watch the error-correcting code repair wrong bits**

```bash
python3 examples/02_ldpc_error_correction.py
```

Starting from a clean 174-bit codeword, the script makes some bits wrongly believed (weakly) and lets belief propagation fix them.
In the run used for this guide, all 20 trials were corrected up to 20 wrong bits, 19 of 20 at 25, and 9 of 20 at 30.
Set `WRONG_STRENGTH` to 3.0 to make the wrong bits confidently wrong, and see how much harder that is.

**Example 3: make a test file for WSJT-X**

```bash
python3 examples/03_write_wav_for_wsjtx.py -18
```

This writes `test_ft8.wav`. In WSJT-X choose File > Open and select it, with the mode set to FT8. Try several SNR
values and compare where WSJT-X still decodes with where this simple receiver stops. The paper explains the gap: WSJT-X
has extra decoding methods (ordered-statistics decoding and others) that this code leaves out.

## 5. Experiments to try

- **Smoothing.** In `ft8.synth()`, change the `bt=2.0` argument. Large values mean almost no smoothing and a spectrum with
  far-out sidebands (Figure 3 of the paper).
- **The Costas array.** Edit the `COSTAS = (3, 1, 4, 0, 6, 5, 2)` line in `ft8.py` to the plain staircase
  `(0, 1, 2, 3, 4, 5, 6)` and run `python3 sim_snr.py 50`. The staircase gives ambiguous sync peaks. In a 60-frame test
  with a Costas search, the real array decoded 60/60 at -17 dB and 47/60 at -19 dB; the staircase decoded 52/60 and 29/60.
  The sample is small, so treat the numbers as indicative. Change the line back afterwards.
- **Belief-propagation effort.** In `ft8dec.decode_at()`, lower `max_iter`. The decode edge moves, as in Figure 9.
- **Look inside a decode.** `r = ft8dec.decode_at(y, 6000, 1000.0)` returns a dictionary. `r['S']` is the 79x8 table of
  tone measurements the decoder works from, `r['llr']` the 174 bit confidences, and `r['trace']` the number of violated
  parity rules after each iteration.

## 6. Reproduce the paper's numbers and figures

```bash
python3 capacity.py          # 8-FSK capacity and the SNR ladder -> capacity.json  (about 10 seconds)
python3 sim_snr.py 250       # decode probability vs SNR -> sim_snr.json (250 frames per point; roughly 4 minutes on 2 cores)
python3 figs_a.py            # signal, waterfall, timing, packing, Costas and sync figures -> figs/ (fig01 to fig09)
python3 figs_b.py            # LDPC, belief propagation, SNR, Shannon and decode-probability figures -> figs/ (fig10 to fig14)
```

The numbers in the PNG file names are not the paper's figure numbers (for example, `fig06_gfsk.png` is Figure 3 in the paper).
`figs_b.py` reads `capacity.json` and `sim_snr.json`.

`sim_snr.py` and `capacity.py` overwrite the stored `.json` results, and the stored results are what the figures in
the paper used. To experiment without losing them, copy the folder first or run `git checkout sim_snr.json capacity.json`
afterwards. For a quick look, `python3 sim_snr.py 50` takes under a minute; its curves are noisier.

**Optional: check the encoder against an independent implementation.** Build [`ft8_lib`](https://github.com/kgoba/ft8_lib)
(`make gen_ft8 decode_ft8` in its folder), then:

```bash
FT8_LIB=/path/to/ft8_lib ./validate_encoder.sh
```

This compares the 79 tones from `ft8.py` against the C encoder for a set of test messages. Every line should say `OK`.

## 7. Troubleshooting

- **`ModuleNotFoundError: No module named 'ft8'`**: run from the repository root, not from inside `examples/`.
- **`FileNotFoundError: ldpc_tables.npz`**: same cause. The file is in the repository root.
- **Matplotlib window or backend errors on a server**: set `MPLBACKEND=Agg`; the figure scripts only write PNG files.
- **A `sim_snr.py` run takes a long time**: lower the frame count (`python3 sim_snr.py 50`), or edit `Pool(2)` to use more cores.

## 8. Limits of this code

It is an educational implementation. It sends and decodes standard messages and 13-character free text; it does not
implement hashed or non-standard callsigns, DXpedition or contest message types, ordered-statistics decoding,
a-priori decoding, multi-symbol detection or signal subtraction, so it is about 1.3 dB less sensitive than WSJT-X.
Noise in all simulations is white and Gaussian, with no fading. Please do not use it on the air.
