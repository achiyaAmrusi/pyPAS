"""Spectral peak search using convolutions."""

import matplotlib.pyplot as plt
import numpy as np
from spectrum import Spectrum
from fit_functions import gaussian_1_dev, gaussian


class PeakFilterError(Exception):
    """Base class for errors in PeakFilter."""


class PeakFilter:
    """An energy-dependent kernel that can be convolved with a spectrum.

    To detect lines, a kernel should have a positive component in the center
    and negative wings to subtract the continuum, e.g., a Gaussian or a boxcar:

    +2|     ┌───┐     |
      |     │   │     |
     0|─┬───┼───┼───┬─|
    -1| └───┘   └───┘ |

    The positive part is meant to detect the peak, and the negative part to
    sample the continuum under the peak.

    The kernel should sum to 0.

    The width of the kernel scales proportionally to the square root
    of the x values (which could be energy, ADC channels, fC of charge
    collected, etc.), with a minimum value set by 'fwhm_at_0'.

    """

    def __init__(self, spectrum: Spectrum, ref_x, ref_fwhm, fwhm_at_0=1.0):
        """Initialize with a reference line position and FWHM in x-values."""
        if ref_x <= 0:
            raise PeakFilterError("Reference x must be positive")
        if ref_fwhm <= 0:
            raise PeakFilterError("Reference FWHM must be positive")
        if fwhm_at_0 < 0:
            raise PeakFilterError("FWHM at 0 must be non-negative")
        self.ref_x = float(ref_x)
        self.ref_fwhm = float(ref_fwhm)
        self.fwhm_at_0 = float(fwhm_at_0)
        self.spectrum = spectrum

    def fwhm(self, channels):
        """Calculate the expected FWHM at the given x value."""
        # f(x)^2 = f0^2 + k x^2
        # f1^2 = f0^2 + k x1^2
        # k = (f1^2 - f0^2) / x1^2
        # f(x)^2 = f0^2 + (f1^2 - f0^2) (x/x1)^2
        f0 = self.fwhm_at_0
        f1 = self.ref_fwhm
        x1 = self.ref_x
        fwhm_sqr = f0**2 + (f1**2 - f0**2) * (channels / x1)
        return np.sqrt(fwhm_sqr)

    def kernel(self, x, y):
        """Generate the kernel for the given x value.
        the kernel i"""
        amplitude = 1
        fwhm = self.fwhm(self.spectrum.channels)
        gaussian_1_dev_left = gaussian_1_dev(self.spectrum.channels[:-1], amplitude, x, fwhm)
        gaussian_1_dev_right = gaussian_1_dev(self.spectrum.channels[1:], amplitude, x, fwhm)
        kernel = gaussian_1_dev_right - gaussian_1_dev_left
        return kernel

    def kernel_matrix(self):
        """Build a matrix of the kernel evaluated at each x value."""
        n_channels = len(self.spectrum.channels) - 1
        kernel_mat = np.zeros((n_channels, n_channels))
        for i, x in enumerate(self.spectrum.channels):
            kernel_mat[:, i] = self.kernel(x, self.spectrum.channels)
        # Taking the negative and positive  parts of the kernel separately in order
        # To normalize the negative and positive parts of the kernel
        # later i need to check the sensitivity of that (pretty sure it's pointless after initial check)
        kern_pos = +1 * kernel_mat.clip(0, np.inf)
        kern_neg = -1 * kernel_mat.clip(-np.inf, 0)
        # normalize negative part to be equal to the positive part
        kern_neg *= kern_pos.sum(axis=0) / kern_neg.sum(axis=0)
        return kern_pos - kern_neg

    def plot_matrix(self):
        """Plot the matrix of kernels evaluated across the x values."""
        n_channels = len(self.spectrum.channels) - 1
        kern_mat = self.kernel_matrix()
        kern_min = kern_mat.min()
        kern_max = kern_mat.max()
        kern_min = min(kern_min, -1 * kern_max)
        kern_max = max(kern_max, -1 * kern_min)

        plt.imshow(
            kern_mat.T[::-1, :],
            cmap=plt.get_cmap("bwr"),
            vmin=kern_min,
            vmax=kern_max,
            extent=[n_channels, 0, 0, n_channels],
        )
        plt.colorbar()
        plt.xlabel("Input x")
        plt.ylabel("Output x")
        plt.gca().set_aspect("equal")

    def convolve(self):
        """Convolve this kernel with the data."""
        kern_mat = self.kernel_matrix()
        kern_mat_pos = +1 * kern_mat.clip(0, np.inf)
        kern_mat_neg = -1 * kern_mat.clip(-np.inf, 0)
        peak_plus_bkg = kern_mat_pos*self.spectrum.counts
        bkg = kern_mat_neg*self.spectrum.counts
        return peak_plus_bkg, bkg


class PeakFinder:
    """Find peaks in a spectrum after convolving it with a kernel."""

    def __init__(self, spectrum: Spectrum, ref_x, ref_fwhm, fwhm_tol=(0.5, 1.5)):
        """Initialize with a spectrum and kernel."""
        self.fwhm_tol = tuple(fwhm_tol)
        self.spectrum = spectrum
        self.peak_filter = PeakFilter(spectrum, ref_x, ref_fwhm, 1.0)
        self._peak_plus_bkg = []
        self._bkg = []
        self.Peaks = []

    def calculate(self, spectrum):
        """Calculate the convolution of the spectrum with the kernel."""
        # calculate the convolution
        peak_plus_bkg, bkg = self.peak_filter.convolve()
        self._peak_plus_bkg = peak_plus_bkg
        self._bkg = bkg

    def add_peak(self, xpeak):
        """Add a peak at xpeak to list if it is not already there."""
        bin_edges = self.spectrum.bin_edges_raw

        xmin = bin_edges.min()
        xmax = bin_edges.max()

        if xpeak < xmin or xpeak > xmax:
            raise PeakFinderError(f"Peak x {xpeak} is outside of range {xmin}-{xmax}")
        is_new_x = True
        for cent in self.centroids:
            if abs(xpeak - cent) <= self.min_sep:
                is_new_x = False
        if is_new_x:
            # estimate FWHM using the second derivative
            # snr(x) = snr(x0) - 0.5 d2snr/dx2(x0) (x-x0)^2
            # 0.5 = 1 - 0.5 d2snr/dx2 (fwhm/2)^2 / snr0
            # 1 = d2snr/dx2 (fwhm/2)^2 / snr0
            # fwhm = 2 sqrt(snr0 / d2snr/dx2)
            xbin = self.spectrum.find_bin_index(xpeak, use_kev=False)
            fwhm0 = self.kernel.fwhm(xpeak)
            bw = self.spectrum.bin_widths_raw[0]
            h = int(max(1, 0.2 * fwhm0 / bw))

            # skip peaks that are too close to the edge
            if (xbin - h < 0) or (xbin + h > len(self.snr) - 1):
                warnings.warn(
                    f"Skipping peak @{xpeak}; too close to the edge of the spectrum",
                    PeakFinderWarning,
                )
                return

            d2 = (
                (1 * self.snr[xbin - h] - 2 * self.snr[xbin] + 1 * self.snr[xbin + h])
                / h**2
                / bw**2
            )
            if d2 >= 0:
                raise PeakFinderError("Second derivative must be negative at peak")
            d2 *= -1
            fwhm = 2 * np.sqrt(self.snr[xbin] / d2)
            # add the peak if it has a similar FWHM to the kernel's FWHM
            if self.fwhm_tol[0] * fwhm0 <= fwhm <= self.fwhm_tol[1] * fwhm0:
                self.centroids.append(xpeak)
                self.snrs.append(self.snr[xbin])
                self.fwhms.append(fwhm)
                self.integrals.append(self._signal[xbin])
                self.backgrounds.append(self._bkg[xbin])
        # sort the peaks by centroid
        self.sort_by(self.centroids)

    def plot(self, facecolor="red", linecolor="red", alpha=0.5, peaks=True):
        """Plot the peak signal-to-noise ratios calculated using the kernel."""
        bin_centers = self.spectrum.bin_centers_raw

        if facecolor is not None:
            plt.fill_between(bin_centers, self.snr, 0, color=facecolor, alpha=alpha)
        if linecolor is not None:
            plt.plot(bin_centers, self.snr, "-", color=linecolor)
        if peaks:
            for cent, snr, fwhm in zip(self.centroids, self.snrs, self.fwhms):
                plt.plot([cent] * 2, [0, snr], "b-", lw=1.5)
                plt.plot(cent, snr, "bo")
                plt.plot(
                    [cent - fwhm / 2, cent + fwhm / 2], [snr / 2] * 2, "b-", lw=1.5
                )
        plt.xlim(0, bin_centers.max())
        plt.ylim(0)
        plt.xlabel("x")
        plt.ylabel("SNR")

    def find_peak(self, xpeak, frac_range=(0.8, 1.2), min_snr=2):
        """Find the highest SNR peak within f0*xpeak and f1*xpeak."""
        bin_edges = self.spectrum.bin_edges_raw
        bin_centers = self.spectrum.bin_centers_raw
        xmin = bin_edges[0]
        xmax = bin_edges[-1]

        if xpeak < xmin or xpeak > xmax:
            raise PeakFinderError(
                f"Guess xpeak {xpeak} is outside of range {xmin}-{xmax}"
            )
        if (
            frac_range[0] < 0
            or frac_range[0] > 1
            or frac_range[1] < 1
            or frac_range[0] > frac_range[1]
        ):
            raise PeakFinderError(
                "Fractional range {}-{} is invalid".format(*frac_range)
            )
        if min_snr < 0:
            raise PeakFinderError(f"Minimum SNR {min_snr:.3f} must be > 0")
        if self.snr.max() < min_snr:
            raise PeakFinderError(
                "SNR threshold is {:.3f} but maximum SNR is {:.3f}".format(
                    min_snr, self.snr.max()
                )
            )
        x0 = frac_range[0] * xpeak
        x1 = frac_range[1] * xpeak
        x_range = (x0 <= bin_edges[:-1]) & (bin_edges[:-1] <= x1)
        peak_snr = self.snr[x_range].max()
        if peak_snr < min_snr:
            raise PeakFinderError(
                f"No peak found in range {x0}-{x1} with SNR > {min_snr}"
            )

        peak_index = np.where((self.snr == peak_snr) & x_range)[0][0]
        peak_x = bin_centers[peak_index]
        self.add_peak(peak_x)
        return peak_x

    def find_peaks(self, xmin=None, xmax=None, min_snr=2, max_num=40, reset=False):
        """Find the highest SNR peaks in the data.

        Parameters
        ----------
        xmin
            Left edge of the x-range that should be scanned for
            peaks. Uses min(x-range) if not given.
        xmax
            Right edge of the x-range that should be scanned for
            peaks. Uses max(x-range) if not given.
        min_snr
            Minium SNR for a peak to be added
        max_num
            Maximum number of peaks to be added
        reset
            If true, reset the already found peaks. Useful when
            changing `min_snr` and calling find_peaks again on the
            same x-range.

        """


        bin_edges = self.spectrum.bin_edges_raw
        bin_centers = self.spectrum.bin_centers_raw

        if xmin is None:
            xmin = bin_edges.min()
        if xmax is None:
            xmax = bin_edges.max()
        if (
            xmin < bin_edges.min()
            or xmin > bin_edges.max()
            or xmax > bin_edges.max()
            or xmax < bin_edges.min()
            or xmin > xmax
        ):
            raise PeakFinderError(f"x-axis range {xmin}-{xmax} is invalid")
        if min_snr < 0:
            raise PeakFinderError(f"Minimum SNR {min_snr:.3f} must be > 0")
        if self.snr.max() < min_snr:
            raise PeakFinderError(
                "SNR threshold is {:.3f} but maximum SNR is {:.3f}".format(
                    min_snr, self.snr.max()
                )
            )
        max_num = int(max_num)
        if max_num < 1:
            raise PeakFinderError(f"Must keep at least 1 peak, not {max_num}")

        # find maxima
        peak = (self.snr[:-2] < self.snr[1:-1]) & (self.snr[1:-1] >= self.snr[2:])
        peak = np.append(False, peak)
        peak = np.append(peak, False)
        # select peaks using SNR and centroid criteria
        peak &= min_snr <= self.snr
        peak &= xmin <= bin_edges[:-1]
        peak &= bin_edges[:-1] <= xmax
        for x in bin_centers[peak]:
            self.add_peak(x)
        # reduce number of centroids to a maximum number max_n of highest SNR
        self.sort_by(np.array(self.snrs))
        self.centroids = self.centroids[-max_num:]
        self.snrs = self.snrs[-max_num:]
        self.fwhms = self.fwhms[-max_num:]
        self.integrals = self.integrals[-max_num:]
        self.backgrounds = self.backgrounds[-max_num:]
        # sort by centroid
        self.sort_by(self.centroids)
        self.peak = peak