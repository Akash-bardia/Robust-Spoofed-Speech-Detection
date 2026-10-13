# -*- coding: utf-8 -*-
"""
Module providing Section 2: Speech Processing Theoretical Foundations & Acoustic Phonetics.
"""

from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

def add_speech_processing_foundations(doc, helpers):
    add_heading_1 = helpers['add_heading_1']
    add_heading_2 = helpers['add_heading_2']
    add_heading_3 = helpers['add_heading_3']
    add_body_p = helpers['add_body_p']
    add_callout = helpers['add_callout']
    add_table_title = helpers['add_table_title']
    set_cell_borders = helpers['set_cell_borders']
    set_cell_shading = helpers['set_cell_shading']
    set_cell_margins = helpers['set_cell_margins']

    add_heading_1(doc, "2. Speech Processing Foundations & Acoustic Phonetics")
    add_body_p(doc,
        "Because this investigation operates at the intersection of statistical signal processing and deep neural representations, evaluating synthetic speech countermeasures requires a rigorous understanding of classical speech acoustics. Speech is a non-stationary, time-varying acoustic wave governed by physical biomechanics, aeroacoustics, and resonant filter dynamics. This section outlines the core mathematical and signal-processing principles foundational to the project."
    )

    # 2.1 SOURCE-FILTER MODEL
    add_heading_2(doc, "2.1. The Source-Filter Model of Speech Production")
    add_body_p(doc,
        "The classical linear acoustic theory of speech production (formulated by Gunnar Fant in 1960) models speech generation as the linear cascade of an excitation source E(z), a vocal tract resonance filter V(z), and a radiation impedance R(z) at the lips:"
    )
    add_callout(doc,
        "Fundamental Equation of Speech Production:",
        "S(z) = E(z) * V(z) * R(z)\n"
        "where:\n"
        "• E(z): Excitation Source (quasi-periodic glottal airflow pulses for voiced sounds; turbulent random noise for unvoiced fricatives).\n"
        "• V(z): Vocal Tract Filter (all-pole acoustic resonator tube of length ~17 cm representing the pharynx, oral cavity, and nasal tract).\n"
        "• R(z): Lip Radiation characteristic, commonly approximated as a first-order differentiator R(z) = 1 - z^(-1) introducing a +6 dB/octave spectral tilt.",
        border_color="2563EB", bg_color="F8FAFC"
    )
    add_body_p(doc,
        "In natural human phonation, vocal fold vibrations produce a fundamental frequency F0 (pitch) with rich harmonic overtones at integer multiples k*F0. The acoustic resonances of the vocal tract tube create prominent spectral energy peaks termed formants (F1, F2, F3, F4). F1 is inversely proportional to tongue height (pharyngeal cavity volume), F2 reflects tongue advancement (oral cavity length), and F3 indicates lip rounding and retroflexion."
    )
    add_body_p(doc,
        "How Synthetic Vocoders Violate the Source-Filter Principle:\n"
        "Conventional parametric vocoders (e.g., STRAIGHT, WORLD, Griffin-Lim) and neural waveform generators (e.g., WaveNet, HiFi-GAN) construct synthetic speech by decoupling pitch tracking from spectral envelope estimation. This process inevitably introduces physical acoustic anomalies:\n"
        "1. Glottal Pulse Smearing: Synthetic models approximate the non-linear glottal flow wave with simplified pulse trains or white Gaussian noise, losing natural pulse-to-pulse pitch perturbation (jitter) and amplitude fluctuation (shimmer).\n"
        "2. Formant Bandwidth Distortions: Vocoders over-smooth or artificially widen formant resonances, resulting in muffled synthetic timbre.\n"
        "3. Phase Incoherence: Human auditory perception is largely phase-deaf to single tones, leading historical vocoders to completely discard phase (minimum-phase assumption). When synthesizing waveforms, these phase inconsistencies generate detectable high-frequency phase cancellations and robotic buzz."
    )

    # 2.2 SHORT-TIME ANALYSIS & WINDOWING
    add_heading_2(doc, "2.2. Short-Time Speech Analysis, Windowing, and Time-Frequency Resolution")
    add_body_p(doc,
        "A continuous speech waveform s(t) is fundamentally non-stationary because articulatory organs continuously move. However, over short intervals of 20 ms to 30 ms (the duration of a single glottal cycle or steady phoneme), the vocal tract geometry remains quasi-stationary. Short-Time Fourier Analysis segments the continuous speech signal into overlapping frames using an analysis window w(n):"
    )
    add_callout(doc,
        "Short-Time Fourier Transform (STFT) Mathematical Formulation:",
        "X(m, omega) = Sum_{n = -inf}^{+inf} [ s(n) * w(n - m*H) * e^(-j * omega * n) ]\n"
        "where m is the frame index, H is the hop size (frame shift), and w(n) is the window function of length N.",
        border_color="0D9488", bg_color="F0FDFA"
    )
    add_body_p(doc,
        "Standard Analysis Parameters in Our Pipeline (16 kHz Audio):\n"
        "• Sampling Rate (fs): 16,000 Hz (Nyquist frequency f_Nyq = 8,000 Hz).\n"
        "• Frame Length: 25.0 ms -> N = 0.025 * 16,000 = 400 samples.\n"
        "• Frame Shift (Hop Size): 10.0 ms -> H = 0.010 * 16,000 = 160 samples (60% frame overlap ensuring smooth temporal continuity).\n"
        "• FFT Resolution: N_FFT = 512 or 1024 points (zero-padded for fine frequency interpolation)."
    )
    add_body_p(doc,
        "Windowing Functions and Spectral Leakage:\n"
        "Abruptly truncating speech with a rectangular window introduces severe spectral leakage due to high sidelobes (-13 dB attenuation). In contrast, our system utilizes a symmetric Hamming window:\n"
        "w(n) = 0.54 - 0.46 * cos(2 * pi * n / (N - 1)), for 0 <= n <= N - 1\n"
        "The Hamming window tapers the frame boundaries smoothly to zero, suppressing the first sidelobe to -43 dB. This prevents spectral leakage from dominant low-frequency formants from contaminating subtle high-frequency vocoder cues."
    )
    add_body_p(doc,
        "The Heisenberg-Gabor Uncertainty Trade-off:\n"
        "Time resolution Delta_t and frequency resolution Delta_f are fundamentally constrained by the uncertainty principle: Delta_t * Delta_f >= 1 / (4 * pi). A narrow analysis window (e.g. 5 ms) yields high temporal resolution (capturing individual pitch pulses and plosive bursts) at the expense of frequency resolution (broadening formant peaks). A wide window (e.g. 100 ms) yields sharp harmonic frequency lines but smears rapid phonetic transitions. The chosen 25 ms frame size optimally balances formant identification with temporal stability."
    )

    # 2.3 LFCC vs MFCC
    add_heading_2(doc, "2.3. Front-End Spectral Features: Mathematical Derivation of LFCC vs. MFCC")
    add_body_p(doc,
        "A central pillar of this project is the comparative analysis and deployment of Linear Frequency Cepstral Coefficients (LFCC) alongside Log-Mel Spectrograms. The complete mathematical extraction pipeline for LFCC comprises seven sequential stages:"
    )
    add_body_p(doc,
        "Step 1: Pre-Emphasis High-Pass Filtering:\n"
        "Natural speech exhibits an intrinsic -6 dB/octave downward spectral tilt caused by glottal airflow decay and lip radiation. To balance the energy spectrum and prevent low frequencies from dominating, speech is pre-emphasized via a first-order FIR difference equation:\n"
        "y(n) = s(n) - alpha * s(n - 1), where alpha in [0.95, 0.97].\n"
        "In the frequency domain, the filter transfer function is H_pre(z) = 1 - alpha * z^(-1), amplifying high frequencies above 1 kHz by +6 dB/octave."
    )
    add_body_p(doc,
        "Step 2: Framing and Hamming Windowing:\n"
        "The pre-emphasized signal y(n) is segmented into 25 ms frames with 10 ms hop size and multiplied by the Hamming window w(n)."
    )
    add_body_p(doc,
        "Step 3: Discrete Fourier Transform (FFT Magnitude Spectrum):\n"
        "Each windowed frame is transformed to the frequency domain via N_FFT = 512-point FFT:\n"
        "|Y(m, k)|^2 = |Sum_{n=0}^{N-1} [ y_m(n) * e^(-j * 2 * pi * k * n / N_FFT) ]|^2, k = 0, ..., N_FFT / 2."
    )
    add_body_p(doc,
        "Step 4: Linear Triangular Filterbank Integration:\n"
        "Unlike MFCC which spaces filter center frequencies logarithmically on the Mel scale, LFCC constructs M = 60 linearly spaced triangular bandpass filters spanning 0 Hz to 8000 Hz:\n"
        "H_m(k) = (k - f_{m-1}) / (f_m - f_{m-1}) for f_{m-1} <= k <= f_m\n"
        "H_m(k) = (f_{m+1} - k) / (f_{m+1} - f_m) for f_m <= k <= f_{m+1}\n"
        "Filter energies are computed by taking the inner product of the power spectrum with each filter:\n"
        "E(m) = Sum_{k} [ |Y(m, k)|^2 * H_m(k) ], m = 1, ..., M."
    )
    add_body_p(doc,
        "Step 5: Logarithmic Power Compression:\n"
        "Logarithmic compression models the human perceived loudness response and converts multiplicative spectral convolution into additive components:\n"
        "S(m) = ln(E(m) + epsilon)."
    )
    add_body_p(doc,
        "Step 6: Discrete Cosine Transform (DCT-II):\n"
        "The log filterbank energies are decorrelated into compact cepstral coefficients via Type-II Discrete Cosine Transform:\n"
        "c_n = Sum_{m=1}^{M} [ S(m) * cos( (pi * n * (m - 0.5)) / M ) ], n = 0, ..., C - 1."
    )
    add_body_p(doc,
        "Step 7: Dynamic Differential Features (Delta and Delta-Delta):\n"
        "Static cepstral coefficients c_n(t) capture instantaneous spectral envelopes. To encode articulatory kinematics and vocal tract transitional velocities, first-order differential (Delta) and second-order acceleration (Delta-Delta) coefficients are computed over a regression window of length 2*Theta + 1:\n"
        "Delta c_n(t) = [ Sum_{theta = 1}^{Theta} theta * (c_n(t + theta) - c_n(t - theta)) ] / [ 2 * Sum_{theta = 1}^{Theta} theta^2 ].\n"
        "Stacking static (20-D) + Delta (20-D) + Delta-Delta (20-D) produces the full 60-dimensional LFCC feature representation."
    )

    # 2.4 WHY LFCC OVER MFCC
    add_heading_2(doc, "2.4. Why LFCC Outperforms MFCC in Speech Anti-Spoofing")
    add_body_p(doc,
        "A critical question frequently encountered in oral examinations is: 'Why is LFCC the standard in speech anti-spoofing, while MFCC dominates Automatic Speech Recognition (ASR)?' Table A provides a direct comparative analysis."
    )

    # Table A
    add_table_title(doc, "A", "Theoretical & Empirical Comparison: LFCC vs. MFCC in Speech Anti-Spoofing")
    t_comp = doc.add_table(rows=6, cols=3)
    t_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_h = ["Feature Dimension", "Mel-Frequency Cepstral Coefficients (MFCC)", "Linear Frequency Cepstral Coefficients (LFCC)"]
    t_d = [
        ["Filter Spacing", "Logarithmic on Mel scale: m = 2595 * log10(1 + f/700). High density at low frequencies (<1 kHz).", "Strictly linear spacing: Delta_f = (fs / 2) / M across entire Nyquist range (0 to 8 kHz)."],
        ["High-Frequency Resolution", "Severely compressed. Few wide filters at high frequencies; fine acoustic details are averaged out.", "Uniform high resolution. Preserves narrow-band artifacts and harmonics up to 8 kHz."],
        ["Original Target Task", "Speech & Speaker Recognition: Designed to mimic the human ear and ignore imperceptible noise.", "Speech Anti-Spoofing & Forensics: Engineered to expose unnatural synthesis artifacts."],
        ["Sensitivity to Vocoder Artifacts", "Poor: Vocoder artifacts (phase mismatches, robotic timbre) occur >3.5 kHz, which MFCC compresses.", "Excellent: Directly captures vocoder ringing, high-frequency harmonic smearing, and GAN checkerboard artifacts."],
        ["Empirical ASVspoof Benchmark", "Typical Eval EER: 22% – 28% (higher error rates due to missed vocoder cues).", "Official ASVspoof baseline & SOTA: 8.09% Dev EER, achieving near-perfect 0.00% EER on vocoder GANs."]
    ]
    for c_idx, h in enumerate(t_h):
        cell = t_comp.cell(0, c_idx)
        cell.text = h
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9.5)
            r.bold = True
        set_cell_borders(cell, top={'val': 'single', 'sz': 12, 'color': '1E293B'}, bottom={'val': 'single', 'sz': 8, 'color': '1E293B'})
        set_cell_shading(cell, "F1F5F9")
        set_cell_margins(cell, 60, 60, 80, 80)

    for r_idx, row in enumerate(t_d):
        for c_idx, val in enumerate(row):
            cell = t_comp.cell(r_idx + 1, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(9.0)
                if c_idx == 0:
                    r.bold = True
            set_cell_borders(cell, bottom={'val': 'single', 'sz': 4, 'color': 'E2E8F0'})
            set_cell_margins(cell, 50, 50, 80, 80)
            if r_idx % 2 == 1:
                set_cell_shading(cell, "F8FAFC")

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # 2.5 LPC & RESIDUAL ANALYSIS
    add_heading_2(doc, "2.5. Linear Predictive Coding (LPC) and Vocal Tract Residual Analysis")
    add_body_p(doc,
        "Linear Predictive Coding (LPC) is one of the most powerful parametric modeling techniques in speech analysis. LPC posits that a speech sample s(n) can be accurately approximated as a linear combination of its past p samples plus an excitation error e(n):"
    )
    add_callout(doc,
        "LPC Prediction Filter & Prediction Error Residual:",
        "s_hat(n) = Sum_{k=1}^{p} [ a_k * s(n - k) ]\n"
        "e(n) = s(n) - s_hat(n) = s(n) - Sum_{k=1}^{p} [ a_k * s(n - k) ]\n"
        "where {a_1, a_2, ..., a_p} are the LPC predictor coefficients and p is the prediction order (typically p = 16 to 20 for 16 kHz speech).",
        border_color="7C3AED", bg_color="FAF5FF"
    )
    add_body_p(doc,
        "The prediction coefficients are computed by minimizing the total squared prediction error E = Sum [ e(n)^2 ] using the autocorrelation method. Setting partial derivatives dE/da_k = 0 yields the Yule-Walker equations:\n"
        "Sum_{k=1}^{p} [ a_k * R(|i - k|) ] = -R(i), for i = 1, ..., p\n"
        "This Toeplitz system is solved in O(p^2) operations via the classical Levinson-Durbin recursion algorithm, simultaneously yielding the reflection coefficients k_i (useful for lattice filtering and stability verification)."
    )
    add_body_p(doc,
        "Significance of the LPC Residual in Speech Forensics:\n"
        "The inverse filter A(z) = 1 - Sum_{k=1}^{p} a_k * z^(-k) strips away the vocal tract formant resonances, leaving behind the excitation source signal e(n). In natural human speech, the residual exhibits distinct physical hallmarks:\n"
        "• Voiced Speech: High-kurtosis, quasi-periodic sharp impulses corresponding to instantaneous glottal closure instants (GCI).\n"
        "• Unvoiced Speech: Uncorrelated Gaussian-like white noise.\n"
        "In contrast, synthetic speech generated by neural vocoders or concatenative synthesis displays smeared, chaotic, or non-impulsive residuals where glottal closure peaks are blurred or over-smoothed. Analyzing the LP residual therefore serves as a potent forensic probe for synthetic artifact detection."
    )

    # 2.6 PITCH & HNR
    add_heading_2(doc, "2.6. Pitch (F0) Dynamics and Harmonics-to-Noise Ratio (HNR)")
    add_body_p(doc,
        "Fundamental frequency F0 reflects the vibration rate of the vocal cords (typically 80–180 Hz for adult males, 160–300 Hz for adult females). F0 tracking is commonly executed via the short-time autocorrelation function:\n"
        "R_x(tau) = Sum_{n=0}^{N - 1 - tau} [ x(n) * x(n + tau) ]\n"
        "The pitch period T0 is identified as the lag tau in [tau_min, tau_max] maximizing R_x(tau), giving F0 = fs / T0."
    )
    add_body_p(doc,
        "Harmonics-to-Noise Ratio (HNR) quantifies the ratio of periodic energy (vocal cord oscillation) to aperiodic noise energy (glottal aspiration and turbulent frication):\n"
        "HNR(dB) = 10 * log10( R_x(T0) / (R_x(0) - R_x(T0)) )\n"
        "Synthetic and converted voices exhibit distinctive prosodic flaws: unnatural F0 step discontinuities across phoneme boundaries, abnormal pitch contour flatness, and lower HNR caused by vocoder phase noise."
    )

    # 2.7 ACOUSTIC CHANNEL & REVERBERATION
    add_heading_2(doc, "2.7. Environmental Acoustic Degradation & Channel Modeling")
    add_body_p(doc,
        "In real-world deployment, speech signals propagate through physical rooms and communication channels before reaching biometric microphones. The received degraded signal y(n) is modeled as a convolutional room impulse response plus additive environmental noise:"
    )
    add_callout(doc,
        "Acoustic Degradation Convolutional Model:",
        "y(n) = s(n) * h(n) + w(n)\n"
        "where:\n"
        "• s(n): Clean anechoic speech signal.\n"
        "• h(n): Room Impulse Response (RIR) characterizing multipath acoustic reflections.\n"
        "• w(n): Additive environmental noise (Gaussian, babble, or HVAC noise).\n"
        "• *: Discrete linear convolution operator.",
        border_color="EA580C", bg_color="FFF7ED"
    )
    add_body_p(doc,
        "Room reverberation is quantified by the reverberation time T60 (the duration required for sound energy density to decay by 60 dB following sound cessation). Sabine's classic formula relates T60 to room geometry and wall absorption:\n"
        "T60 = 0.161 * V / A_eff\n"
        "where V is room volume in cubic meters and A_eff is total effective sound absorption surface area."
    )
    add_body_p(doc,
        "Acoustic Impact on Biometric Countermeasures:\n"
        "1. Early Reflections (<50 ms): Reinforce direct sound and alter formant peak heights (comb-filtering effect).\n"
        "2. Late Reverberation (>50 ms): Creates temporal smearing, filling silent phonemic gaps and masking delicate vocoder phase cues.\n"
        "3. Additive Noise: Masks high-frequency linear spectral bins, causing clean baseline detectors to experience catastrophic performance collapse. Hardening the model through acoustic data augmentation is essential for real-world reliability."
    )

    print("Section 2: Speech Processing Foundations added successfully.")

