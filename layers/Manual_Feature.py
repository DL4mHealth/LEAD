import torch
import torch.fft as fft
import torch.nn.functional as F


def bandpass_filter(signal, fs, lowcut, highcut):
    # length of signal
    fft_len = signal.size(1)
    # FFT
    fft_spectrum = torch.fft.rfft(signal, n=fft_len, dim=1)
    # get frequency bins
    freqs = torch.fft.rfftfreq(fft_len, d=1/fs)
    # create mask for freqs
    mask = (freqs >= lowcut) & (freqs <= highcut)
    # expand mask to match fft_spectrum
    mask = mask.view(1, -1, 1).expand_as(fft_spectrum)
    mask = mask.to(signal.device)
    # apply mask
    fft_spectrum = fft_spectrum * mask
    # IFFT
    filtered_signal = torch.fft.irfft(fft_spectrum, n=fft_len, dim=1)

    return filtered_signal


def statistical_feature_extractor(x_enc):
    """
    Extract features from a frequency band of a batch of trials. Map batch with shape BxTxC to Bx(F*C).
    B is the batch size, T is timestamps, C is channels, F is statistical features.
    Statistical features names: Mean, Variance, Skewness, Kurtosis,
    Standard Deviation, Interquartile Range, Maximum, Minimum, Average, Median

    Parameters:
      x_enc : Tensor
        A batch of samples with shape BxTxC

    Returns:
      output : Tensor
        A batch of features with shape Bx10xC
    """
    epsilon = 1e-9  # Small constant to prevent division by zero or log of zero

    # Compute basic statistics
    mean = torch.mean(x_enc, dim=1)
    min = torch.min(x_enc, dim=1).values
    max = torch.max(x_enc, dim=1).values
    std = torch.std(x_enc, dim=1, unbiased=False)  # Set unbiased=False for population std
    var = torch.var(x_enc, dim=1, unbiased=False)  # Set unbiased=False for population variance
    median = torch.median(x_enc, dim=1).values

    # Compute IQR
    q75 = torch.quantile(x_enc, 0.75, dim=1, keepdim=True)
    q25 = torch.quantile(x_enc, 0.25, dim=1, keepdim=True)
    iqr = (q75 - q25).squeeze(1)

    # Calculate the 3rd and 4th moments for skewness and kurtosis
    deviations = x_enc - mean.unsqueeze(1)
    m3 = torch.mean(deviations ** 3, dim=1)
    m2 = std ** 2 + epsilon  # Variance, ensure non-zero with epsilon
    skewness = m3 / (m2 ** 1.5 + epsilon)  # Ensure non-zero denominator

    m4 = torch.mean(deviations ** 4, dim=1)
    kurtosis = m4 / (m2 ** 2 + epsilon) - 3  # Excess kurtosis, ensure non-zero denominator

    # Stack features along a new dimension, 10 features
    output = torch.stack((mean, var, skewness, kurtosis, std, iqr, max, min, mean, median), dim=1)

    return output


def compute_psd(x_enc, fs=128):
    fft_result = torch.fft.rfft(x_enc, dim=1)
    psd = torch.abs(fft_result) ** 2
    return psd


def compute_band_power(psd, fs, band):
    freqs = torch.linspace(0, fs / 2, steps=psd.shape[1])  # Adjusted to match PSD shape
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    # print(band_mask)
    band_power = torch.sum(psd[:, band_mask, :], dim=1)
    return band_power


def power_feature_extractor(x_enc, fs):
    """
    Extract features from a frequency band of a batch of trials. Map batch with shape BxTxC to BxFxC.
    B is the batch size, T is timestamps, C is channels
    Calculate Power Spectral Density and some PSD related biomarkers

    Parameters:
      x_enc : Tensor
        A batch of samples with shape BxTxC
      fs : int
        Sampling frequency rate

    Returns:
      output : Tensor
        A batch of psd with shape Bx(11)xC
    """
    epsilon = 1e-9  # Small constant to prevent division by zero or log of zero

    # Frequency domain features (PSD)
    psd = compute_psd(x_enc, fs)  # Shape: [B, F, C]

    # Compute band powers for α and β bands
    delta_power = compute_band_power(psd, fs, (0.5, 4))
    theta_power = compute_band_power(psd, fs, (4, 8))
    alpha_power = compute_band_power(psd, fs, (8, 12))
    beta_power = compute_band_power(psd, fs, (12, 30))

    # Compute total power for normalization (to compute relative power)
    total_power = torch.sum(psd, dim=1) + epsilon

    # Compute relative power for bands
    delta_rel_power = delta_power / total_power
    theta_rel_power = theta_power / total_power
    alpha_rel_power = alpha_power / total_power
    beta_rel_power = beta_power / total_power

    # Compute ratios of EEG rhythms
    theta_alpha_ratio = theta_power / (alpha_power + epsilon)
    alpha_beta_ratio = alpha_power / (beta_power + epsilon)

    # 11 features
    output = torch.stack((delta_power, theta_power, alpha_power, beta_power, total_power,
                          theta_alpha_ratio, alpha_beta_ratio,
                          delta_rel_power, theta_rel_power, alpha_rel_power, beta_rel_power), dim=1)

    return output


def _hilbert(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute analytic signal (Hilbert transform) along `dim`
    using FFT – compatible with PyTorch < 2.3.

    Args
    ----
    x   : (..., T, ...) real-valued tensor
    dim : dimension along which to apply Hilbert transform

    Returns
    -------
    analytic : complex tensor with same shape as `x`
    """
    N = x.size(dim)
    Xf = fft.fft(x, dim=dim)                 # full FFT (complex)

    # Construct the frequency-domain multiplier h
    h = torch.zeros(N, dtype=x.dtype, device=x.device)
    if N % 2 == 0:                           # even length
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:                                    # odd length
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    # Reshape h for broadcasting along `dim`
    shape = [1] * x.ndim
    shape[dim] = N
    h = h.view(shape)

    analytic = fft.ifft(Xf * h, dim=dim)     # IFFT gives analytic signal
    return analytic


def spectral_feature_extractor(x_enc: torch.Tensor, fs: float, rolloff: float = 0.85) -> torch.Tensor:
    """
    Pure-Torch implementation of spectral features.
    Args
    ----
    x_enc : (B, T, C) real-valued tensor      – EEG batch
    fs : float                     – sampling rate (Hz)
    rolloff : float = 0.85                    – roll-off ratio for spectral-rolloff

    Returns
    -------
    out : (B, 7, C) tensor on same device as x_enc
    """
    B, T, C = x_enc.shape
    device = x_enc.device
    dtype = x_enc.dtype

    # ---------- FFT ----------
    X = fft.rfft(x_enc, dim=1)                # (B, F, C)  F = T//2 + 1
    mag = X.abs()                             # magnitude spectrum
    phase = torch.angle(X)                    # phase spectrum
    F_bins = mag.shape[1]

    # Pre-compute frequency vector (F, )   – broadcast later
    freqs = fft.rfftfreq(T, d=1.0 / fs).to(device=device, dtype=dtype)

    # ---------- 1. Phase Coherence ----------
    # PC = |mean(e^{jφ})|
    phase_coh = torch.exp(1j * phase).mean(dim=1).abs()          # (B, C)

    # ---------- 2. Spectral Centroid ----------
    # Σ f·|X| / Σ |X|
    num = (freqs[:, None] * mag).sum(dim=1)                      # (B, C)
    den = mag.sum(dim=1) + 1e-8
    spec_centroid = num / den                                    # (B, C)

    # ---------- 3. Spectral Roll-off ----------
    # lowest bin where cumulative energy ≥ rolloff·total
    cumsum_mag = mag.cumsum(dim=1)                               # (B, F, C)
    thresh = rolloff * cumsum_mag[:, -1, :]                      # (B, C)
    # boolean mask of first bin crossing threshold
    rolloff_idx = (cumsum_mag >= thresh[:, None, :]).float().argmax(dim=1)  # (B, C)
    spec_rolloff = freqs[rolloff_idx]                            # (B, C)

    # ---------- 4. Spectral Peak ----------
    spec_peak, _ = mag.max(dim=1)                                # (B, C)

    # ---------- 5. Average Magnitude ----------
    avg_mag = mag.mean(dim=1)                                    # (B, C)

    # ---------- 6. Median Frequency ----------
    half_energy = 0.5 * cumsum_mag[:, -1, :]                     # (B, C)
    med_idx = (cumsum_mag >= half_energy[:, None, :]).float().argmax(dim=1)  # (B, C)
    med_freq = freqs[med_idx]                                    # (B, C)

    # ---------- 7. Amplitude Modulation ----------
    # std of Hilbert envelope  -> use torch.abs(FFT^{-1}{X} + j*H(X))
    analytic = _hilbert(x_enc, dim=1)                         # (B, T, C) complex
    amp_env = analytic.abs()                                    # envelope
    amp_mod = amp_env.std(dim=1)                                # (B, C)

    # ---------- Stack features ----------
    output = torch.stack([
        phase_coh,
        spec_centroid,
        spec_rolloff,
        spec_peak,
        avg_mag,
        med_freq,
        amp_mod
    ], dim=1)                                                    # (B, 7, C)

    return output


def cal_shannon_entropy(psd):
    """
    Calculate Shannon Entropy from PSD.
    """
    epsilon = 1e-9
    norm_psd = psd / (psd.sum(dim=1, keepdim=True) + epsilon)
    log_psd = torch.log2(norm_psd + epsilon)
    entropy = -torch.sum(norm_psd * log_psd, dim=1)
    return entropy


def cal_tsallis_entropy(signal, q=2):
    epsilon = 1e-9
    probabilities = signal / (signal.sum(dim=1, keepdim=True) + epsilon)
    tsallis_en = (1 - torch.pow(probabilities, q).sum(dim=1)) / (q - 1)
    return tsallis_en


def cal_spectral_entropy(psd):
    epsilon = 1e-9
    psd_norm = psd / (torch.sum(psd, dim=1, keepdim=True) + epsilon)
    spectral_entropy = -torch.sum(psd_norm * torch.log(psd_norm + epsilon), dim=1)
    return spectral_entropy


def complexity_feature_extractor(x_enc, fs):
    """
    Extract specified entropy features from a frequency band of a batch of trials.
    Map batch with shape BxTxC to BxFxC.
    B is the batch size, T is timestamps, C is channels

    Parameters:
      x_enc : Tensor
        A batch of samples with shape BxTxC
      fs : int
        Sampling frequency rate

    Returns:
      output : Tensor
        A batch of entropy features with shape BxFxC
    """

    epsilon = 1e-9  # Small constant to prevent division by zero or log of zero

    # Frequency domain features (PSD)
    psd = compute_psd(x_enc, fs=fs)
    psd_norm = psd / (torch.sum(psd, dim=1, keepdim=True) + epsilon)  # Normalize PSD for probability distribution

    # Spectral Entropy
    spectral_entropy = cal_spectral_entropy(psd)

    # Shannon entropy
    shannon_entropy = cal_shannon_entropy(psd)

    # Tsallis entropy
    tsallis_entropy = cal_tsallis_entropy(psd, q=2)

    output = torch.stack((spectral_entropy, tsallis_entropy, shannon_entropy), dim=1)

    return output


def feature_extractor(x_enc, fs=128):
    """
    Extract features from a batch of trials. Map batch with shape BxTxC to Bx(F*C).
    B is the batch size, T is timestamps, C is channels, F is features.
    Features are statistical and power features.

    Parameters:
      x_enc : Tensor
        A batch of samples with shape BxTxC
      fs : int
        Sampling frequency rate

    Returns:
      output : Tensor
        A batch of features with shape Bx(F*C)
    """
    # Extract statistical features
    stat_features = statistical_feature_extractor(x_enc)  # Shape: [B, 10, C]

    # Extract power features
    power_features = power_feature_extractor(x_enc, fs)  # Shape: [B, 11, C]

    # Extract spectral features
    spectral_features = spectral_feature_extractor(x_enc, fs)  # Shape: [B, 7, C]

    # Extract complexity features
    complexity_features = complexity_feature_extractor(x_enc, fs)  # Shape: [B, 3, C]

    # Concatenate along the feature dimension
    output = torch.cat((stat_features, power_features, spectral_features, complexity_features), dim=1)  # Shape: [B, (10+11+7+3)*C]

    return output
