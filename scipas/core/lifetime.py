from scispectrum import Spectrum
from scipas.core.time_resolution import TimeResolution, TIME_AXIS_NAME


class PASLifetime:
    """
    Represents a lifetime spectrum with a resolution instance.

    Parameters
    ----------
    lifetime : Spectrum
        Counts per bin against time. Its axis must be calibrated to time, i.e.
        carry the axis name "time" (TIME_AXIS_NAME), in ns.
    resolution : TimeResolution
        Instrument response function of the measurement.

    Raises
    ------
    ValueError
        If the spectrum axis is not named "time". The axis name is the only
        record of what the axis measures, so a spectrum calibrated to energy
        is rejected here rather than being fitted as if it were time.

    Notes
    -----
    Read the axis values as "pals.lifetime.axis". That attribute does not
    depend on the axis name, so it is the accessor to use throughout.
    """

    def __init__(self, lifetime: Spectrum, resolution: TimeResolution):
        if lifetime.axis_name != TIME_AXIS_NAME:
            raise ValueError(
                f"The lifetime spectrum axis is named '{lifetime.axis_name}', "
                f"expected '{TIME_AXIS_NAME}'. Calibrate it with "
                f"AxisCalibration(..., name='{TIME_AXIS_NAME}') so the axis "
                f"states that it measures time."
            )
        self.lifetime = lifetime
        self.resolution = resolution
