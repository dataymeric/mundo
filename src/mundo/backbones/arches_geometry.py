"""Time-axis padding of the ArchesWeather network."""


def time_tokens(L: int) -> tuple[int, int]:
    """Pad `L` frames on the time axis for the ArchesWeather Swin.

    Args:
        L: Number of frames in the window.

    Returns:
        z: Number of time tokens.
        pad_after: Zero frames appended after the `L` frames.
    """
    z = -(-(L + 2) // 2)
    if (z + 1) % 2:
        z += 1
    return z, 2 * z - L - 1
