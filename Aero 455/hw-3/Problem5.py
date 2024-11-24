import numpy as np
import matplotlib.pyplot as plt
from Problem2 import (
    TipLossFactor,
    TipLossFunction,
)

number_of_mistakes_i_am_willing_to_get_that_grade_otherwise_i_will_q_drop_off_of_rudder_tower_jk_this_project_is_actually_kinda_fun_after_you_consume_enough_caffiene = (
    103
)
delta_radius = (
    1
    / number_of_mistakes_i_am_willing_to_get_that_grade_otherwise_i_will_q_drop_off_of_rudder_tower_jk_this_project_is_actually_kinda_fun_after_you_consume_enough_caffiene
)
starting_radius = 0.2
coefficient_lift_alpha = 2 * np.pi
coefficient_thrust_required = 0.008


def Solidity(normalized_radius: float, number_of_blades: int, taper: float) -> float:
    root_chord = 0.1 * np.pi / number_of_blades
    return (number_of_blades / (np.pi * normalized_radius)) * (
        root_chord - root_chord * (1 - 1 / taper) * normalized_radius
    )


def InflowWithTipLoss(
    normalized_radius: float,
    number_of_blades: int,
    tip_loss: float,
    twist: float,
    taper: float,
) -> float:
    solidity = Solidity(normalized_radius, number_of_blades, taper)
    coefficient_lift_alpha = 2 * np.pi
    first_term = (solidity * coefficient_lift_alpha) / (16 * tip_loss)
    second_term = (
        np.sqrt(
            1
            + ((32 * tip_loss) / (solidity * coefficient_lift_alpha))
            * twist
            * normalized_radius
        )
        - 1
    )
    return first_term * second_term


def ObjectiveInflowFunction(
    normalized_radius: float,
    inflow_ratio: float,
    tip_loss: float,
    twist: float,
    number_of_blades: int,
    taper: float,
) -> float:
    """
    Checks how close the values for inflow, tip_loss, and twist are to zero

    Returns:
    discrete value of the governing equation for this problem

    """
    solidity = Solidity(normalized_radius, number_of_blades, taper)
    coefficient_lift_alpha = 2 * np.pi

    return (
        inflow_ratio**2
        + ((solidity * coefficient_lift_alpha) / (8 * tip_loss)) * inflow_ratio
        - ((solidity * coefficient_lift_alpha) / (8 * tip_loss))
        * twist
        * normalized_radius
    )


def LinearTwist(
    normalized_radius: float,
    linear_twist_rate_rads: float,
    number_of_blades: int,
    taper: float,
) -> float:
    """
    ### MAKE SURE YOUR TWIST RATE IS IN RADIANS YOU BAFFOON OF A STUDENT

    Returns:
    :cat:
    ![maew]("./hw-3/appendix/maew.png")
    """
    solidity = Solidity(normalized_radius, number_of_blades, taper)
    coefficient_lift_alpha = 2 * np.pi
    coefficient_thrust_required = 0.008
    twist_0 = (
        (6 * coefficient_thrust_required) / (solidity * coefficient_lift_alpha)
        - 0.75 * linear_twist_rate_rads
        + 1.5 * np.sqrt(coefficient_thrust_required / 2)
    )
    return twist_0 + linear_twist_rate_rads * normalized_radius


def InflowBEMT(
    normalized_radius: float,
    twist: float,
    number_of_blades: int,
    taper: float,
) -> float:
    """
    ## Finds the inflow at given span of the blade as a function of the radius, number of blades, linear twist, and taper

    Returns:
    Inflow Ratio
    """
    twist_i = LinearTwist(normalized_radius, twist, number_of_blades, taper)
    inflow_i = InflowWithTipLoss(normalized_radius, number_of_blades, 1, twist_i, taper)
    tip_factor_i = TipLossFactor(normalized_radius, number_of_blades, inflow_i)
    tip_function_i = TipLossFunction(tip_factor_i)
    iteration_count = 0
    iteration_max = 30
    tolerance = 1e-6
    objfunc_i = ObjectiveInflowFunction(
        normalized_radius, inflow_i, tip_function_i, twist_i, number_of_blades, taper
    )
    while abs(objfunc_i) >= tolerance:
        iteration_count += 1
        inflow_i = InflowWithTipLoss(
            normalized_radius, number_of_blades, tip_function_i, twist_i, taper
        )
        tip_factor_i = TipLossFactor(normalized_radius, number_of_blades, inflow_i)
        tip_function_i = TipLossFunction(tip_factor_i)
        objfunc_i = ObjectiveInflowFunction(
            normalized_radius,
            inflow_i,
            tip_function_i,
            twist_i,
            number_of_blades,
            taper,
        )
        if iteration_count == iteration_max:
            print("Scheme didn't converge")
            print([inflow_i, tip_function_i, tip_factor_i, abs(objfunc_i)])
            return inflow_i

    return inflow_i


def PlotProblem5():
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(10, 12))

    radius_values = np.arange(starting_radius, 1, delta_radius)
    fixed_twist = np.deg2rad(-15)
    taper_ratio_values = [1, 2, 3]
    # Plot the first subplot
    for taper_ratio in taper_ratio_values:
        axis.plot(
            radius_values,
            [InflowBEMT(r, fixed_twist, 2, taper_ratio) for r in radius_values],
            label=r"$\pi_{taper} = $" + f"{round(taper_ratio,0)}",
            linestyle="-",
        )

    # First plot formatting
    axis.set_title("Inflow Ratio")
    axis.legend()
    axis.set_xlabel(r"$\lambda$")
    axis.set_ylabel(r"$r = \frac{y}{R}$")
    # # Second plot formatting
    # axis[0, 1].set_title("Power Factor")
    # axis[0, 1].legend()
    # axis[0, 1].set_xlabel(r"$C_T$")
    # axis[0, 1].set_ylabel(r"$\kappa$")
    # # Third plot formatting
    # axis[1, 0].set_title("Figure of Merit")
    # axis[1, 0].legend()
    # axis[1, 0].set_xlabel(r"$C_T$")
    # axis[1, 0].set_ylabel(r"$FM$")

    # Adjust layout to prevent overlap between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()

    return None


def Example3DPlot():
    # Generate some random data for x, y coordinates
    x = np.random.rand(100)
    y = np.random.rand(100)

    # Generate some random z values
    z = np.sin(x) + 2 * np.cos(y)

    # Create a figure and an axis for the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the data as a scatter plot
    scat = ax.scatter(x, y, z, c=z, cmap="viridis", marker="o")

    # Set labels for the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Rotate the plot to see a specific plane
    # For example, rotate around the x-axis by 30 degrees and around the z-axis by 45 degrees
    ax.view_init(elev=30, azim=45)

    # Add a color bar to show the mapping of colors to values
    fig.colorbar(scat, shrink=0.5)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    PlotProblem5()
