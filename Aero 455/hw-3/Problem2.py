import matplotlib.pyplot as plt
import numpy as np

number_of_mistakes = 100
delta_radius = 1 / number_of_mistakes
starting_radius = 0.2
solidity = 0.1
coefficient_lift_alpha = 2 * np.pi
coefficient_thrust_required = 0.008
no_twist = (
    (4 * coefficient_thrust_required) / (solidity * coefficient_lift_alpha)
) + np.sqrt(coefficient_thrust_required / 2)


def ObjectiveFunction(
    normalized_radius: float, inflow_ratio: float, tip_loss: float, twist: float
) -> float:
    """
    Checks how close the values for inflow, tip_loss, and twist are to zero

    Returns:
    discrete value of the governing equation for this problem

    """
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi

    return (
        inflow_ratio**2
        + ((solidity * coefficient_lift_alpha) / (8 * tip_loss)) * inflow_ratio
        - ((solidity * coefficient_lift_alpha) / (8 * tip_loss))
        * twist
        * normalized_radius
    )


def InflowWithTipLoss(normalized_radius: float, tip_loss: float, twist: float) -> float:
    solidity = 0.1
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


def TipLossFunction(tip_loss_factor: float) -> float:
    return (2 / np.pi) * np.arccos(np.exp(-tip_loss_factor))


def TipLossFactor(
    normalized_radius: float, number_of_blades: int, inflow_ratio: float
) -> float:
    return (number_of_blades / 2) * ((1 - normalized_radius) / inflow_ratio)


def NumbericalBEMT(normalized_radius: float, number_of_blades: int) -> list:
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    coefficient_thrust_required = 0.008
    no_twist = (
        (4 * coefficient_thrust_required) / (solidity * coefficient_lift_alpha)
    ) + np.sqrt(coefficient_thrust_required / 2)
    inflow_i = InflowWithTipLoss(normalized_radius, 1, no_twist)
    tip_factor_i = TipLossFactor(normalized_radius, number_of_blades, inflow_i)
    tip_function_i = TipLossFunction(tip_factor_i)
    iteration_count = 0
    iteration_max = 30
    tolerance = 1e-6
    objfunc_i = ObjectiveFunction(normalized_radius, inflow_i, tip_function_i, no_twist)
    while abs(objfunc_i) >= tolerance:
        iteration_count += 1
        inflow_i = InflowWithTipLoss(normalized_radius, tip_function_i, no_twist)
        tip_factor_i = TipLossFactor(normalized_radius, number_of_blades, inflow_i)
        tip_function_i = TipLossFunction(tip_factor_i)
        objfunc_i = ObjectiveFunction(
            normalized_radius, inflow_i, tip_function_i, no_twist
        )
        if iteration_count == iteration_max:
            print("Scheme didn't converge")
            return [inflow_i, tip_function_i, tip_factor_i]

    return [inflow_i, tip_function_i, tip_factor_i]


def DeltaCoefficientThrustBEMT(
    normalized_radius: float, number_of_blades: int
) -> float:
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    coefficient_thrust_required = 0.008
    no_twist = (
        (4 * coefficient_thrust_required) / (solidity * coefficient_lift_alpha)
    ) + np.sqrt(coefficient_thrust_required / 2)
    return (solidity * coefficient_lift_alpha / 2) * (
        no_twist * normalized_radius**2
        - NumbericalBEMT(normalized_radius, number_of_blades)[0] * normalized_radius
    )


def DeltaCoefficientTorqueBEMT(
    normalized_radius: float, number_of_blades: int
) -> float:
    solidity = 0.1
    coefficient_drag_0 = 0.008
    inflow = NumbericalBEMT(normalized_radius, number_of_blades)[0]
    delta_CT = DeltaCoefficientThrustBEMT(normalized_radius, number_of_blades)
    return (
        inflow * delta_CT + 0.5 * solidity * coefficient_drag_0 * normalized_radius**3
    )


def CoefficientLiftBEMT(normalized_radius: float, number_of_blades: int) -> float:
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    coefficient_thrust_required = 0.008
    no_twist = (
        (4 * coefficient_thrust_required) / (solidity * coefficient_lift_alpha)
    ) + np.sqrt(coefficient_thrust_required / 2)
    inflow = NumbericalBEMT(normalized_radius, number_of_blades)[0]
    return coefficient_lift_alpha * (no_twist - inflow / normalized_radius)


def PowerFactor(number_of_blades: int) -> float:
    delta_CPi = []
    delta_CT_list = []
    for r_n in np.arange(starting_radius, 1, delta_radius):
        if number_of_blades > 0:
            inflow = NumbericalBEMT(r_n, number_of_blades)[0]
            delta_CT = DeltaCoefficientThrustBEMT(r_n, number_of_blades) * delta_radius
            delta_CT_list.append(delta_CT)
            delta_CPi.append(inflow * delta_CT)
        else:
            inflow = ((solidity * coefficient_lift_alpha) / 16) * (
                np.sqrt(1 + (32 / (solidity * coefficient_lift_alpha)) * no_twist * r_n)
                - 1
            )
            delta_CT = (
                (solidity * coefficient_lift_alpha / 2)
                * (no_twist * r_n**2 - inflow * r_n)
                * delta_radius
            )
            delta_CT_list.append(delta_CT)
            delta_CPi.append(inflow * delta_CT)
    return np.sum(delta_CPi) / (np.sum(delta_CT_list) ** 1.5 / 2**0.5)


def PlotProblem2():
    fig, axis = plt.subplots(ncols=2, nrows=2, figsize=(10, 12))

    number_of_mistakes = 100
    delta_radius = 1 / number_of_mistakes
    starting_radius = 0.1
    radius_values = np.arange(starting_radius, 1, delta_radius)
    coefficient_drag_0 = 0.008
    # Plot the first subplot
    two_blade_inflow_values = [NumbericalBEMT(r, 2)[0] for r in radius_values]
    four_blade_inflow_values = [NumbericalBEMT(r, 4)[0] for r in radius_values]
    InflowNoTipLoss = lambda normalized_radius: (
        (solidity * coefficient_lift_alpha) / 16
    ) * (
        np.sqrt(
            1
            + (32 / (solidity * coefficient_lift_alpha)) * no_twist * normalized_radius
        )
        - 1
    )
    axis[0, 0].plot(
        radius_values, [InflowNoTipLoss(r) for r in radius_values], label="No Tip Loss"
    )
    axis[0, 0].plot(
        radius_values, two_blade_inflow_values, label=r"$N_b = 2$", linestyle="--"
    )
    axis[0, 0].plot(
        radius_values, four_blade_inflow_values, label=r"$N_b = 4$", linestyle=":"
    )
    axis[0, 0].set_title("Inflow Ratio")
    axis[0, 0].legend()
    axis[0, 0].set_xlabel("r")
    axis[0, 0].set_ylabel(r"$\lambda$")

    # # Plot the second subplot
    DeltaCTNoLoss = lambda r: (solidity * coefficient_lift_alpha / 2) * (
        no_twist * r**2 - InflowNoTipLoss(r) * r
    )
    axis[0, 1].plot(
        radius_values,
        [DeltaCTNoLoss(r) for r in radius_values],
        label="No Tip Loss",
        linestyle="-",
    )
    axis[0, 1].plot(
        radius_values,
        [DeltaCoefficientThrustBEMT(r, 2) for r in radius_values],
        label=r"$N_b = 2$",
        linestyle="--",
    )
    axis[0, 1].plot(
        radius_values,
        [DeltaCoefficientThrustBEMT(r, 4) for r in radius_values],
        label=r"$N_b = 4$",
        linestyle=":",
    )
    axis[0, 1].legend()
    axis[0, 1].set_xlabel("r")
    axis[0, 1].set_ylabel(r"$\frac{\Delta{C}_T}{\Delta{r}}$")
    axis[0, 1].set_title("Change in Coefficient of Thrust")

    # Plot the third subplot
    axis[1, 0].plot(
        radius_values,
        [
            InflowNoTipLoss(r) * DeltaCTNoLoss(r)
            + 0.5 * solidity * coefficient_drag_0 * r**3
            for r in radius_values
        ],
        label="No Tip Loss",
        linestyle="-",
    )
    axis[1, 0].plot(
        radius_values,
        [DeltaCoefficientTorqueBEMT(r, 2) for r in radius_values],
        label=r"$N_b = 2$",
        linestyle="--",
    )
    axis[1, 0].plot(
        radius_values,
        [DeltaCoefficientTorqueBEMT(r, 4) for r in radius_values],
        label=r"$N_b = 4$",
        linestyle=":",
    )
    axis[1, 0].legend()
    axis[1, 0].set_xlabel("r")
    axis[1, 0].set_ylabel(r"$\frac{\Delta{C}_Q}{\Delta{r}}$")
    axis[1, 0].set_title("Change in Coefficient of Torque")

    # 4th subplot
    CoefficientLiftNoLoss = lambda r: coefficient_lift_alpha * (
        no_twist - InflowNoTipLoss(r) / r
    )
    axis[1, 1].plot(
        radius_values,
        [CoefficientLiftNoLoss(r) for r in radius_values],
        label="No Tip Loss",
        linestyle="-",
    )
    axis[1, 1].plot(
        radius_values,
        [CoefficientLiftBEMT(r, 2) for r in radius_values],
        label=r"$N_b = 2$",
        linestyle="--",
    )
    axis[1, 1].plot(
        radius_values,
        [CoefficientLiftBEMT(r, 4) for r in radius_values],
        label=r"$N_b = 4$",
        linestyle=":",
    )
    axis[1, 1].legend()
    axis[1, 1].set_xlabel("r")
    axis[1, 1].set_ylabel(r"$C_l$")
    axis[1, 1].set_title("Change in Coefficient of Lift")

    # Adjust layout to prevent overlap between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()

    return None


power_factor_no_tip_loss = PowerFactor(0)
power_factor_two_blades = PowerFactor(2)
power_factor_four_blades = PowerFactor(4)

if __name__ == "__main__":
    PlotProblem2()
