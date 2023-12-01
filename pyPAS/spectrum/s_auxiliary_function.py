from uncertainties import unumpy


def residual_std_weight(params, data_x, data_y):
    a = params['a']
    b = params['b']
    return (a*data_x + b - unumpy.nominal_values(data_y)) / (unumpy.std_devs(data_y)+1e-5)
