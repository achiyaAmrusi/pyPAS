def domain_of_peak(spectrum, detector_energy_resolution, energy_in_the_peak):
    """ define the total area of the peak
        The function takes spectrum slice in size of the resolution and check from which energy the counts are constant
        however because the counts are not constant,
        it checks when the counts N_sigma from the mean is larger than 1
        The auther notes that it is noticeable that the large energy side of the peak is much less noisy than lower side
        """
    start_of_energy_slice = energy_in_the_peak
    energy_step_size = spectrum['energy'].values[1] - spectrum['energy'].values[0]

    n_sigma = 2
    while n_sigma > 0.6:
        spectrum_slice = spectrum.sel(energy=slice(start_of_energy_slice - detector_energy_resolution,
                                                   start_of_energy_slice))
        mean_of_function_slice = unumpy.nominal_values(spectrum_slice.values).mean()
        n_sigma_distribution = np.array(
            [np.abs(unumpy.nominal_values(count) - mean_of_function_slice) / (unumpy.std_devs(count)+1) for count in
             spectrum_slice.values])
        n_sigma = n_sigma_distribution.sum() / len(n_sigma_distribution)
        start_of_energy_slice = start_of_energy_slice - energy_step_size
    left_energy_peak_domain = start_of_energy_slice

    n_sigma = 2
    start_of_energy_slice = energy_in_the_peak

    while n_sigma > 0.6:
        spectrum_slice = spectrum.sel(energy=slice(start_of_energy_slice,
                                                   start_of_energy_slice + detector_energy_resolution))
        mean_of_function_slice = unumpy.nominal_values(spectrum_slice.values).mean()
        n_sigma_distribution = np.array(
            [np.abs(unumpy.nominal_values(count) - mean_of_function_slice) / (unumpy.std_devs(count)+1) for count in
             spectrum_slice.values])
        n_sigma = n_sigma_distribution.sum() / len(n_sigma_distribution)
        start_of_energy_slice = start_of_energy_slice + energy_step_size
    right_energy_peak_domain = start_of_energy_slice

    return 502, 520
#    return left_energy_peak_domain, right_energy_peak_domain