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


def PlotProblem3():
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(12, 17))

    radius_values = np.arange(starting_radius, 1, delta_radius)
    linear_twist_rate_values = np.deg2rad(np.linspace(-20, 0, 6))
    coefficient_thrust_max = []
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
        axis.plot(
            coefficient_thrust_values,
            [CoefficientPowerHover(r, 4, theta_tw) for r in radius_values],
            label=r"$\theta_{tw} = $" + f"{round(np.degrees(theta_tw),0)}",
            linestyle="-",
        )
    axis.set_title("Coefficient Power of Hover Required")
    axis.legend()
    axis.set_xlabel(r"$C_T$")
    axis.set_ylabel(r"$C_P$")

    # Adjust layout to prevent overlap between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()

    return None


if __name__ == "__main__":
    PlotProblem3()
