# -*- coding: utf-8 -*-
"""
Module providing Section 11: Oral Defense & Viva Examination Guide (Speech Processing Q&A).
"""

def add_viva_qa_section(doc, helpers):
    add_heading_1 = helpers['add_heading_1']
    add_body_p = helpers['add_body_p']
    add_qa_item = helpers['add_qa_item']

    add_heading_1(doc, "11. Speech Processing Oral Defense & Viva Examination Guide")
    add_body_p(doc,
        "This section prepares the project team for academic defense, oral viva examinations, and faculty questioning on speech processing principles. Each entry provides a comprehensive, technically rigorous answer grounded in speech acoustics, digital signal processing, and empirical biometrics."
    )

    # Q1: SOURCE-FILTER
    add_qa_item(doc, 1,
        "Explain the Source-Filter model of speech production. How do synthetic speech engines and neural vocoders violate this model?",
        "Fant's Source-Filter theory models speech generation as S(z) = E(z) * V(z) * R(z). An excitation source E(z) (quasi-periodic glottal pulses for voiced sounds, turbulent friction for unvoiced sounds) passes through an acoustic all-pole vocal tract filter V(z) with resonant formant frequencies (F1, F2, F3), radiating through the lips R(z) as a first-order differentiator (+6 dB/octave tilt).\n"
        "Synthetic vocoders (WORLD, STRAIGHT, Griffin-Lim) and neural waveform generators (WaveNet, HiFi-GAN) violate this physical model in three distinct ways:\n"
        "1. Decoupled Pitch & Envelope: They estimate F0 and spectral envelopes independently, introducing phase discontinuities and non-physical glottal flow transitions.\n"
        "2. Formant Bandwidth Broadening: Vocoders over-smooth resonance peaks, reducing the natural sharpness of formant bandwidths.\n"
        "3. Phase Incoherence: Vocoders frequently discard phase information (relying on minimum-phase assumptions). During synthesis, the reconstructed phase exhibits unnatural high-frequency cancellation patterns that do not occur in human vocal fold physiology."
    )

    # Q2: LFCC VS MFCC
    add_qa_item(doc, 2,
        "Why is LFCC strongly preferred over MFCC for synthetic speech detection, whereas MFCC is universally preferred for Automatic Speech Recognition (ASR)?",
        "The fundamental divergence stems from the design objectives of the two tasks:\n"
        "• ASR aims to recognize linguistic phonemes regardless of speaker or acoustic channel. It employs the non-linear Mel scale (m = 2595 * log10(1 + f/700)), which matches human auditory perception by placing dense filterbanks at low frequencies (<1 kHz) and sparse, widely spaced filterbanks at high frequencies (>3 kHz). This deliberately discards high-frequency spectral nuances as perceptual noise.\n"
        "• Speech Anti-Spoofing aims to detect unnatural acoustic artifacts left by synthesis algorithms. Vocoder artifacts (phase mismatches, high-frequency spectral ringing, GAN checkerboard patterns, and neural waveform discontinuities) predominantly reside in the upper linear spectrum (between 3.5 kHz and 8.0 kHz). Because MFCC compresses and averages out these high frequencies, it throws away the primary forensic evidence! LFCC employs linearly spaced triangular filters across the entire Nyquist band, maintaining equal resolution in high frequencies and directly exposing vocoder fingerprints."
    )

    # Q3: PRE-EMPHASIS
    add_qa_item(doc, 3,
        "What is the mathematical formulation and acoustic rationale of pre-emphasis filtering: y(n) = s(n) - alpha * s(n - 1)?",
        "Pre-emphasis is a first-order FIR high-pass filter with transfer function H(z) = 1 - alpha * z^(-1), where alpha is typically 0.95 to 0.97. The acoustic necessity arises from two physical phenomena:\n"
        "1. Glottal Flow Spectrum Decay: Natural vocal fold oscillation exhibits an intrinsic -12 dB/octave downward spectral tilt.\n"
        "2. Lip Radiation: Lip opening introduces a +6 dB/octave upward differentiation tilt. Combining these results in a net -6 dB/octave attenuation of speech energy as frequency increases.\n"
        "Without pre-emphasis, high-frequency formants (F3, F4) and subtle synthesis cues possess negligible energy compared to the dominant F1 formant and fundamental frequency F0. Pre-emphasis boosts high frequencies by approximately +6 dB/octave above 1 kHz, flattening the global speech spectrum and preventing numerical underflow in downstream filterbank integration."
    )

    # Q4: WINDOWING & STFT
    add_qa_item(doc, 4,
        "Why do we segment speech into 25 ms frames with a 10 ms hop size using a Hamming window? Explain the time-frequency trade-off.",
        "Speech is non-stationary because articulatory organs continuously move. However, over 20 to 30 ms intervals, the physical configuration of the vocal tract remains quasi-stationary. Therefore, 25 ms (400 samples at 16 kHz) is chosen as the standard analysis frame.\n"
        "• Frame Shift (10 ms / 160 samples): A 10 ms hop yields a 60% overlap between adjacent frames. This ensures temporal continuity, smooth parameter interpolation, and prevents transient phonemes (such as plosive release bursts) from being missed at frame boundaries.\n"
        "• Hamming Window: Rectangular windowing causes severe spectral leakage with high sidelobes (-13 dB). The Hamming window smoothly tapers frame edges to zero, suppressing the first sidelobe to -43 dB and preventing energy from dominant lower formants from spilling into high-frequency filterbanks.\n"
        "• Time-Frequency Trade-off (Heisenberg-Gabor): Delta_t * Delta_f >= 1 / (4 * pi). A narrow window provides high time resolution but blurs frequency formants; a wide window provides fine harmonic lines but smears rapid phoneme transitions. The 25 ms frame length optimally balances formant envelope extraction with articulatory stability."
    )

    # Q5: DYNAMIC DELTAS
    add_qa_item(doc, 5,
        "What physical phenomena do dynamic cepstral features (Delta and Delta-Delta coefficients) capture in speech processing?",
        "Static cepstral coefficients c_n(t) capture instantaneous vocal tract shapes within a single 25 ms frame. However, human speech is inherently dynamic; co-articulation causes formants to smoothly transition between phonemes as the tongue and lips move.\n"
        "• Delta Coefficients (Velocity): The first temporal derivative Delta c_n(t) measures the rate of articulatory movement (formant transition velocity). It is computed via orthogonal polynomial regression across a temporal window of +/-Theta frames.\n"
        "• Delta-Delta Coefficients (Acceleration): The second temporal derivative Delta^2 c_n(t) measures articulatory acceleration and muscular effort.\n"
        "In synthetic speech, concatenative algorithms introduce unnatural velocity step jumps at stitch boundaries, while neural vocoders often produce over-smoothed, sluggish dynamic trajectories. By appending Delta and Delta-Delta coefficients (expanding 20 static coefficients to 60-D), our detector models both instantaneous acoustics and physical articulatory kinematics."
    )

    # Q6: LPC & RESIDUAL
    add_qa_item(doc, 6,
        "Explain Linear Predictive Coding (LPC). Why is the prediction residual e(n) particularly valuable for speech forensics?",
        "LPC models speech as an all-pole linear filter: s(n) = Sum_{k=1}^p [ a_k * s(n - k) ] + e(n). By minimizing the mean squared prediction error via the Levinson-Durbin algorithm, LPC estimates the vocal tract resonance filter V(z) = 1 / A(z).\n"
        "The inverse filter A(z) removes the vocal tract formants from the speech signal, outputting the prediction residual e(n) = s(n) - s_hat(n), which represents the glottal excitation source. The LP residual provides profound forensic evidence:\n"
        "• In bona fide speech, voiced frames exhibit sharp, distinct, high-kurtosis glottal pulse spikes corresponding to vocal fold closure instants (GCIs).\n"
        "• In synthetic speech, parametric vocoders approximate excitation with periodic pulse trains or white noise, resulting in blurred, chaotic, or non-impulsive residuals that easily trigger countermeasure alarms."
    )

    # Q7: REVERBERATION & T60
    add_qa_item(doc, 7,
        "How do room reverberation (T60) and additive noise degrade speech anti-spoofing countermeasures mathematically?",
        "In acoustic environments, degraded speech follows the convolutional model y(n) = s(n) * h(n) + w(n), where h(n) is the Room Impulse Response (RIR) and w(n) is additive noise.\n"
        "• Early Reflections (<50 ms): Arrive immediately after the direct path, causing comb-filtering in the frequency domain that alters formant amplitudes and shifts apparent cepstral peaks.\n"
        "• Late Reverberation (>50 ms, characterized by T60): Sound energy echoes off walls, creating an exponentially decaying tail that fills phonemic silence and blurs high-frequency temporal transitions. This masks the subtle phase and harmonic discontinuities that anti-spoofing detectors rely on.\n"
        "• Additive Noise: Flat noise floors elevate the energy of high-frequency filterbanks, drowning out vocoder ringing. Without acoustic data augmentation, clean models interpret noise as non-vocoded human speech, resulting in severe performance degradation (as demonstrated by our baseline collapsing to 0.4653 AUC under 5 dB noise)."
    )

    # Q8: MFM ACTIVATION
    add_qa_item(doc, 8,
        "What is the Max-Feature-Map (MFM) activation function and why does it outperform standard ReLU in speech feature networks?",
        "Max-Feature-Map (MFM) is a competitive non-linear activation function introduced by Wu et al. Specifically, given an intermediate feature map x with 2*C channels, MFM splits the channels into two halves and computes the element-wise maximum: MFM(x) = max(x_{1:C}, x_{C+1:2C}), reducing the channels to C.\n"
        "MFM offers three distinct advantages over standard ReLU for speech spectrogram processing:\n"
        "1. Competitive Feature Selection: MFM acts as an adaptive filter that dynamically selects between two competing candidate representations, separating genuine speech cues from background noise.\n"
        "2. Avoids Dead Neurons: Unlike ReLU, which zeroes out all negative values and causes dead gradients, MFM always propagates gradients through whichever half possessed the larger activation.\n"
        "3. Compact Embeddings: MFM forces the network to learn complementary feature representations in the two halves, yielding compact, high-density embeddings (our entire LCNN requires only 87K parameters)."
    )

    # Q9: ACOUSTIC COLLAPSE
    add_qa_item(doc, 9,
        "Why did the clean baseline model collapse under 5 dB Gaussian noise (AUC 0.4653, EER 54.80%), and how did data augmentation prevent this failure?",
        "The clean baseline was trained strictly on anechoic, studio-recorded audio from ASVspoof 2019 LA. Consequently, the network associated clean high frequencies with genuine human voice and learned to detect synthetic speech based exclusively on subtle spectral artifacts.\n"
        "When exposed to heavy 5 dB noise, the Gaussian noise floor completely overwhelmed the high-frequency linear LFCC filterbanks. The clean network suffered from distribution shift and inverted its decision boundary, producing an AUC of 0.4653 (worse than random coin flips at 0.50) and an EER of 54.80%.\n"
        "Our Augmented MFM-LCNN was trained with probabilistic on-the-fly additive noise (10–25 dB SNR), synthetic room impulse responses, and 2D SpecAugment. This forced the convolutional kernels to learn invariant spectro-temporal features that are resilient to noise floors, retaining 0.6606 AUC under 5 dB noise (+0.1953 lift) and 0.8855 AUC under reverberation."
    )

    # Q10: EER VS ACCURACY
    add_qa_item(doc, 10,
        "Why is Equal Error Rate (EER) the primary evaluation metric in biometrics rather than Accuracy, particularly on the ASVspoof 2019 LA Evaluation set?",
        "Classification accuracy is notoriously deceptive under severe class imbalance. The ASVspoof 2019 Evaluation partition contains 63,882 spoofed utterances but only 7,355 bona fide utterances (an 8.7:1 ratio; spoof comprises 89.68% of the data). A naive, non-functional dummy classifier that unconditionally labels every utterance as 'spoof' would automatically achieve 89.68% Accuracy while having zero real-world biometric utility!\n"
        "Equal Error Rate (EER) resolves this by evaluating the threshold-independent trade-off between two operational error rates:\n"
        "• False Acceptance Rate (FAR): The probability that a spoofed voice is mistakenly accepted as genuine human speech (security vulnerability).\n"
        "• False Rejection Rate (FRR): The probability that a genuine human user is rejected (user convenience degradation).\n"
        "EER represents the exact operating point where FAR equals FRR. It provides an objective, distribution-independent benchmark of biometric countermeasure accuracy that cannot be inflated by class imbalance."
    )

    print("Section 11: Viva Examination Guide added successfully.")
