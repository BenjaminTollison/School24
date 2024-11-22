import matplotlib.pyplot as plt
import numpy as np
from Problem2 import (
    ObjectiveFunction,
    InflowWithTipLoss,
    TipLossFactor,
    TipLossFunction,
)

number_of_mistakes_i_am_willing_to_get_that_grade = 101
delta_radius = 1 / number_of_mistakes_i_am_willing_to_get_that_grade
starting_radius = 0.2
solidity = 0.1
coefficient_lift_alpha = 2 * np.pi
coefficient_thrust_required = 0.008
tip_twist = (
    (4 * coefficient_thrust_required) / (solidity * coefficient_lift_alpha)
) + np.sqrt(coefficient_thrust_required / 2)


def LinearTwist(normalized_radius: float, linear_twist_rate_rads: float) -> float:
    """
    ### MAKE SURE YOUR TWIST RATE IS IN RADIANS YOU BAFFOON OF A STUDENT

    Returns:
    :cat:
    ![maew]("./hw-3/appendix/maew.png")
    """
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    coefficient_thrust_required = 0.008
    twist_0 = (
        (6 * coefficient_thrust_required) / (solidity * coefficient_lift_alpha)
        - 0.75 * linear_twist_rate_rads
        + 1.5 * np.sqrt(coefficient_thrust_required / 2)
    )
    return twist_0 + linear_twist_rate_rads * normalized_radius


def NumbericalBEMT(
    normalized_radius: float, number_of_blades: int, linear_twist_rate_rads: float
) -> float:
    """
    ## Finds the inflow at given span of the blade as a function of the radius, number of blades, and linear twist

    Returns:
    Inflow Ratio
    """
    twist_i = LinearTwist(normalized_radius, linear_twist_rate_rads)
    inflow_i = InflowWithTipLoss(normalized_radius, 1, twist_i)
    tip_factor_i = TipLossFactor(normalized_radius, number_of_blades, inflow_i)
    tip_function_i = TipLossFunction(tip_factor_i)
    iteration_count = 0
    iteration_max = 30
    tolerance = 1e-6
    objfunc_i = ObjectiveFunction(normalized_radius, inflow_i, tip_function_i, twist_i)
    while abs(objfunc_i) >= tolerance:
        iteration_count += 1
        inflow_i = InflowWithTipLoss(normalized_radius, tip_function_i, twist_i)
        tip_factor_i = TipLossFactor(normalized_radius, number_of_blades, inflow_i)
        tip_function_i = TipLossFunction(tip_factor_i)
        objfunc_i = ObjectiveFunction(
            normalized_radius, inflow_i, tip_function_i, twist_i
        )
        if iteration_count == iteration_max:
            print("Scheme didn't converge")
            print([inflow_i, tip_function_i, tip_factor_i, abs(objfunc_i)])
            return inflow_i

    return inflow_i


def DeltaCoefficientThrustBEMT(
    normalized_radius: float, number_of_blades: int, linear_twist_rate_rads: float
) -> float:
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    twist = LinearTwist(normalized_radius, linear_twist_rate_rads)
    inflow = NumbericalBEMT(normalized_radius, number_of_blades, twist)
    return (solidity * coefficient_lift_alpha / 2) * (
        twist * normalized_radius**2 - inflow * normalized_radius
    )


def DeltaCoefficientPowerInducedBEMT(
    normalized_radius: float, number_of_blades: int, linear_twist_rate_rads: float
) -> float:
    inflow = NumbericalBEMT(normalized_radius, number_of_blades, linear_twist_rate_rads)
    delta_CT = DeltaCoefficientThrustBEMT(
        normalized_radius, number_of_blades, linear_twist_rate_rads
    )
    return inflow * delta_CT


def DeltaCoefficientPowerProfileBEMT(nomalized_radius: float) -> float:
    solidity = 0.1
    coefficient_drag_given = 0.011
    return 0.5 * solidity * coefficient_drag_given * nomalized_radius**3


def CoefficientLiftBEMT(
    normalized_radius: float, number_of_blades: int, linear_twist_rate_rads: float
) -> float:
    coefficient_lift_alpha = 2 * np.pi
    twist = LinearTwist(normalized_radius, linear_twist_rate_rads)
    inflow = NumbericalBEMT(normalized_radius, number_of_blades, linear_twist_rate_rads)
    return coefficient_lift_alpha * (twist - inflow / normalized_radius)


def PlotProblem3():
    fig, axis = plt.subplots(ncols=2, nrows=3, figsize=(12, 17))

    radius_values = np.arange(starting_radius, 1, delta_radius)
    linear_twist_rate_values = np.deg2rad(np.linspace(-30, 10, 9))
    coefficient_drag_0 = 0.008
    # Plot the first subplot
    for theta_tw in linear_twist_rate_values:
        axis[0, 0].plot(
            radius_values,
            [NumbericalBEMT(r, 4, theta_tw) for r in radius_values],
            label=r"$\theta_{tw} = $" + f"{round(np.degrees(theta_tw),0)}",
            linestyle="-",
        )
        axis[1, 0].plot(
            radius_values,
            [DeltaCoefficientThrustBEMT(r, 4, theta_tw) for r in radius_values],
            label=r"$\theta_{tw} = $" + f"{round(np.degrees(theta_tw),0)}",
            linestyle="-",
        )
        axis[0, 1].plot(
            radius_values,
            [DeltaCoefficientPowerInducedBEMT(r, 4, theta_tw) for r in radius_values],
            label=r"$\theta_{tw} = $" + f"{round(np.degrees(theta_tw),0)}",
            linestyle="-",
        )
        axis[2, 0].plot(
            radius_values,
            [CoefficientLiftBEMT(r, 4, theta_tw) for r in radius_values],
            label=r"$\theta_{tw} = $" + f"{round(np.degrees(theta_tw),0)}",
            linestyle="-",
        )
    axis[1, 1].plot(
        radius_values,
        [DeltaCoefficientPowerProfileBEMT(r) for r in radius_values],
        linestyle="-",
    )

    axis[0, 0].set_title("Inflow Ratio")
    axis[0, 0].legend()
    axis[0, 0].set_xlabel("r")
    axis[0, 0].set_ylabel(r"$\lambda$")
    # figure 2 formating
    axis[0, 1].set_title("Change in Coefficient of Thrust")
    axis[0, 1].legend()
    axis[0, 1].set_xlabel("r")
    axis[0, 1].set_ylabel(r"$\frac{dC_T}{dr}$")
    # figure 3 formating
    axis[1, 0].set_title("Change in Coefficient of Induced Power")
    axis[1, 0].legend()
    axis[1, 0].set_xlabel("r")
    axis[1, 0].set_ylabel(r"$\frac{dC_{Pi}}{dr}$")
    # figure 4 formating
    axis[1, 1].set_title("Change in Coefficient of Profile Power")
    axis[1, 1].set_xlabel("r")
    axis[1, 1].set_ylabel(r"$\frac{dC_{Pp}}{dr}$")
    # figure 5 formating
    axis[2, 0].set_title("Coefficient of Lift")
    axis[2, 0].legend()
    axis[2, 0].set_xlabel("r")
    axis[2, 0].set_ylabel(r"$C_l$")

    # Adjust layout to prevent overlap between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()

    return None


if __name__ == "__main__":
    PlotProblem3()
