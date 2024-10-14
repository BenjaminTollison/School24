import numpy as np

atmospheric_table = {
    "0": [1.225, 7.249],
    "25": [3.899 * 10**-2, 6.349],
    "30": [1.774 * 10**-2, 6.6882],
    "40": [3.972 * 10**-3, 7.554],
    "50": [1.057 * 10**-3, 8.382],
    "60": [3.206 * 10**-4, 7.714],
    "70": [8.770 * 10**-5, 6.549],
    "80": [1.905 * 10**-5, 5.799],
    "90": [3.396 * 10**-6, 5.382],
    "100": [5.297 * 10**-7, 5.877],
    "110": [9.661 * 10**-8, 7.263],
    "120": [2.438 * 10**-8, 9.473],
    "130": [8.484 * 10**-9, 12.636],
    "140": [3.845 * 10**-9, 16.149],
    "150": [2.070 * 10**-9, 22.523],
    "180": [5.464 * 10**-10, 27.740],
    "200": [2.789 * 10**-10, 37.105],
    "250": [7.248 * 10**-11, 45.546],
    "300": [2.148 * 10**-11, 53.628],
    "350": [9.518 * 10**-12, 53.298],
    "400": [3.725 * 10**-12, 58.515],
    "450": [1.585 * 10**-12, 60.828],
    "500": [6.967 * 10**-13, 63.822],
    "600": [1.454 * 10**-13, 71.835],
    "700": [3.614 * 10**-14, 88.667],
    "800": [1.170 * 10**-14, 124.64],
    "900": [5.245 * 10**-15, 181.05],
    "1000": [3.019 * 10**-15, 268],
}


def Density(altitude):
    base_altitudes = [int(i) for i in atmospheric_table.keys()]
    if altitude < min(base_altitudes):
        return atmospheric_table[str(min(base_altitudes))][
            0
        ]  # Return the density at the minimum altitude
    elif altitude > max(base_altitudes):
        return atmospheric_table[str(max(base_altitudes))][
            0
        ]  # Return the density at the maximum altitude

    # Find the surrounding altitudes
    sorted_list = sorted(base_altitudes)
    for i in range(len(sorted_list) - 1):
        if sorted_list[i] <= altitude < sorted_list[i + 1]:
            lower_altitude = sorted_list[i]
            higher_altitude = sorted_list[i + 1]
            break
    else:
        lower_altitude, higher_altitude = sorted_list[-2], sorted_list[-1]  # Edge case

    lower_altitude_key = str(lower_altitude)
    higher_altitude_key = str(higher_altitude)

    # Extract nominal densities and scale heights
    nominal_density, scale_height = atmospheric_table[lower_altitude_key][:2]
    return nominal_density * np.exp((lower_altitude - altitude) / scale_height)


def LinearScaleHeight(altitude):
    base_altitudes = [int(i) for i in atmospheric_table.keys()]
    if altitude < min(base_altitudes):
        return atmospheric_table[str(min(base_altitudes))][
            1
        ]  # Return the scale height at the minimum altitude
    elif altitude > max(base_altitudes):
        return atmospheric_table[str(max(base_altitudes))][
            1
        ]  # Return the scale height at the maximum altitude

    # Find the surrounding altitudes
    sorted_list = sorted(base_altitudes)
    for i in range(len(sorted_list) - 1):
        if sorted_list[i] <= altitude < sorted_list[i + 1]:
            lower_altitude = sorted_list[i]
            higher_altitude = sorted_list[i + 1]
            break
    else:
        lower_altitude, higher_altitude = sorted_list[-2], sorted_list[-1]  # Edge case

    lower_altitude_key = str(lower_altitude)
    higher_altitude_key = str(higher_altitude)

    # Extract scale heights for interpolation
    _, lower_scale_height = atmospheric_table[lower_altitude_key][:2]
    _, upper_scale_height = atmospheric_table[higher_altitude_key][:2]

    # Linear interpolation for scale height
    return ((altitude - lower_altitude) / (higher_altitude - lower_altitude)) * (
        upper_scale_height - lower_scale_height
    ) + lower_scale_height
