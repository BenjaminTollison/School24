import matplotlib.pyplot as plt
import numpy as np
from Problem3 import LinearTwist, NumbericalBEMT, DeltaCoefficientThrustBEMT

number_of_mistakes_i_am_willing_to_get_that_grade_otherwise_i_will_q_drop_off_of_rudder_tower = (
    102
)
delta_radius = (
    1
    / number_of_mistakes_i_am_willing_to_get_that_grade_otherwise_i_will_q_drop_off_of_rudder_tower
)
starting_radius = 0.2
solidity = 0.1
coefficient_lift_alpha = 2 * np.pi
coefficient_thrust_required = 0.008


def CoefficientDrag(
    normalized_radius: float, number_of_blades: float, linear_twist_rate_rads: float
) -> float:
    twist = LinearTwist(normalized_radius, linear_twist_rate_rads)
    inflow = NumbericalBEMT(normalized_radius, number_of_blades, linear_twist_rate_rads)
    alpha = twist - inflow / normalized_radius
    return 0.011 - 0.025 * alpha + 0.65 * alpha**2


def DeltaCoefficientPowerHover(
    normalized_radius: float, number_of_blades: float, linear_twist_rate_rads: float
) -> float:
    solidity = 0.1
    number_of_mistakes_i_am_willing_to_get_that_grade_otherwise_i_will_q_drop_off_of_rudder_tower = (
        102
    )
    delta_radius = (
        1
        / number_of_mistakes_i_am_willing_to_get_that_grade_otherwise_i_will_q_drop_off_of_rudder_tower
    )
    inflow = NumbericalBEMT(normalized_radius, number_of_blades, linear_twist_rate_rads)
    delta_CT = DeltaCoefficientThrustBEMT(
        normalized_radius, number_of_blades, linear_twist_rate_rads
    )
    coefficient_drag = CoefficientDrag(
        normalized_radius, number_of_blades, linear_twist_rate_rads
    )
    return (
        inflow * delta_CT
        + 0.5 * solidity * coefficient_drag * normalized_radius**3 * delta_radius
    )


def CoefficientPowerHover(
    normalized_radius: float, number_of_blades: float, linear_twist_rate_rads: float
) -> float:
    number_of_mistakes_i_am_willing_to_get_that_grade_otherwise_i_will_q_drop_off_of_rudder_tower = (
        102
    )
    delta_radius = (
        1
        / number_of_mistakes_i_am_willing_to_get_that_grade_otherwise_i_will_q_drop_off_of_rudder_tower
    )
    starting_radius = 0.2
    return np.sum(
        [
            DeltaCoefficientPowerHover(r, number_of_blades, linear_twist_rate_rads)
            for r in np.arange(starting_radius, normalized_radius, delta_radius)
        ]
    )


def DeltaCoefficientPowerInduced(
    normalized_radius: float, number_of_blades: float, linear_twist_rate_rads: float
) -> float:
    inflow = NumbericalBEMT(normalized_radius, number_of_blades, linear_twist_rate_rads)
    delta_CT = (
        DeltaCoefficientThrustBEMT(
            normalized_radius, number_of_blades, linear_twist_rate_rads
        )
        * delta_radius
    )
    return inflow * delta_CT


def CoefficientPowerInduced(
    normalized_radius: float, number_of_blades: float, linear_twist_rate_rads: float
) -> float:
    delta_CP_list = [
        DeltaCoefficientPowerInduced(r, number_of_blades, linear_twist_rate_rads)
        for r in np.arange(starting_radius, normalized_radius, delta_radius)
    ]
    return np.sum(delta_CP_list)


def CoefficientThrust(
    normalized_radius: float, number_of_blades: float, linear_twist_rate_rads: float
) -> float:
    delta_CT_list = [
        DeltaCoefficientThrustBEMT(r, number_of_blades, linear_twist_rate_rads)
        * delta_radius
        for r in np.arange(starting_radius, normalized_radius, delta_radius)
    ]
    return np.sum(delta_CT_list)


def PowerFactor(
    normalized_radius: float, number_of_blades: float, linear_twist_rate_rads: float
) -> float:
    Cp_i = CoefficientPowerInduced(
        normalized_radius, number_of_blades, linear_twist_rate_rads
    )
    Ct = CoefficientThrust(normalized_radius, number_of_blades, linear_twist_rate_rads)
    try:
        power_factor = Cp_i / (Ct**1.5 / 2**0.5)
        if power_factor >= 3:
            return np.nan
        else:
            return power_factor
    except RuntimeWarning:
        return np.nan


def FigureOfMerit(
    normalized_radius: float, number_of_blades: float, linear_twist_rate_rads: float
) -> float:
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    coefficient_thrust = CoefficientThrust(
        normalized_radius, number_of_blades, linear_twist_rate_rads
    )
    power_factor = PowerFactor(
        normalized_radius, number_of_blades, linear_twist_rate_rads
    )
    cp_ideal = coefficient_thrust**1.5 / 2**0.5
    cp_induced = power_factor * coefficient_thrust**1.5 / 2**0.5
    cp_profile = (
        solidity * 0.011 / 8
        - (2 * 0.025 / (3 * coefficient_lift_alpha)) * coefficient_thrust
        + (4 * 0.65 / (solidity * coefficient_lift_alpha**2) * coefficient_thrust**2)
    )
    return cp_ideal / (cp_induced + cp_profile)


def PlotProblem4():
    fig, axis = plt.subplots(ncols=2, nrows=2, figsize=(10, 12))

    radius_values = np.arange(starting_radius, 1, delta_radius)
    linear_twist_rate_values = np.deg2rad(np.linspace(-20, 0, 6))
    coefficient_thrust_max = []
    point_styles = [
        ".",  # dot
        ",",  # pixel (point)
        "o",  # circle
        "s",  # square
        "p",  # pentagon
        "h",  # hexagon
        "H",  # rotated hexagon
        "+",  # plus sign
        "x",  # cross
        "D",  # diamond
        "d",  # thin diamond
        "|",  # vertical line
        "_",  # horizontal line
    ]
    # Plot the first subplot
    for theta_tw in linear_twist_rate_values:
        coefficient_thrust_max.append(
            np.sum([DeltaCoefficientThrustBEMT(r, 4, theta_tw) for r in radius_values])
        )
    coefficient_thrust_max = np.max(coefficient_thrust_max)
    coefficient_thrust_values = np.linspace(
        0.0, coefficient_thrust_max, len(radius_values)
    )
    for theta_tw in linear_twist_rate_values:
        axis[0, 0].plot(
            coefficient_thrust_values,
            [CoefficientPowerHover(r, 4, theta_tw) for r in radius_values],
            label=r"$\theta_{tw} = $" + f"{round(np.degrees(theta_tw),0)}",
            linestyle="-",
        )
        # Second plot
        Ct_temp_values = [CoefficientThrust(r, 4, theta_tw) for r in radius_values]
        power_factor_temp_values = [PowerFactor(r, 4, theta_tw) for r in radius_values]
        theta_tw_index = -np.argmin(
            np.abs(linear_twist_rate_values - np.array(theta_tw))
        )
        point_marker = point_styles[theta_tw_index]
        axis[0, 1].scatter(
            Ct_temp_values,
            power_factor_temp_values,
            marker=point_marker,
            label=r"$\theta_{tw} = $" + f"{round(np.degrees(theta_tw),0)}",
        )
        # Third plot
        FM_temp_values = [FigureOfMerit(r, 4, theta_tw) for r in radius_values]
        axis[1, 0].scatter(
            Ct_temp_values,
            FM_temp_values,
            marker=point_marker,
            label=r"$\theta_{tw} = $" + f"{round(np.degrees(theta_tw),0)}",
        )

    # First plot formatting
    axis[0, 0].set_title("Coefficient Power of Hover Required")
    axis[0, 0].legend()
    axis[0, 0].set_xlabel(r"$C_T$")
    axis[0, 0].set_ylabel(r"$C_P$")
    # Second plot formatting
    axis[0, 1].set_title("Power Factor")
    axis[0, 1].legend()
    axis[0, 1].set_xlabel(r"$C_T$")
    axis[0, 1].set_ylabel(r"$\kappa$")
    # Third plot formatting
    axis[1, 0].set_title("Figure of Merit")
    axis[1, 0].legend()
    axis[1, 0].set_xlabel(r"$C_T$")
    axis[1, 0].set_ylabel(r"$FM$")

    # Adjust layout to prevent overlap between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()

    return None


if __name__ == "__main__":
    PlotProblem4()
