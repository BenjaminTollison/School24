import matplotlib.pyplot as plt
import numpy as np


def Twist(normalized_radius: float, tip_twist: float) -> float:
    return tip_twist / normalized_radius


def ObjectiveFunction(inflow: float, twist: float, normalized_radius: float) -> float:
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    return (
        inflow**2
        + (solidity * coefficient_lift_alpha / 8) * inflow
        - (solidity * coefficient_lift_alpha / 8) * twist * normalized_radius
    )


def FindInflowAndTipTwist(normalized_radius: float) -> list:
    """
    This function finds the inflow ratio and the tip twist at a specific radius with BEMT

    Returns:
    list: [inflow_i,tip_twist_i]
    """
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    coefficient_thrust_given = 0.008
    inflow_i = np.sqrt(coefficient_thrust_given / 2)
    tip_twist_i = 4 * coefficient_thrust_given / (solidity * coefficient_lift_alpha)
    twist_i = Twist(normalized_radius, tip_twist_i)
    objfunc_i = ObjectiveFunction(inflow_i, twist_i, normalized_radius)
    iteration_count = 0
    max_iteration = 30
    tolerance = 1e-6
    while abs(objfunc_i) >= tolerance:
        iteration_count += 1
        tip_twist_i = (
            4 * coefficient_thrust_given / (solidity * coefficient_lift_alpha)
            + inflow_i
        )
        twist_i = Twist(normalized_radius, tip_twist_i)
        inflow_i = (solidity * coefficient_lift_alpha / 16) * (
            np.sqrt(
                1
                + (32 / (solidity * coefficient_lift_alpha))
                * twist_i
                * normalized_radius
            )
            - 1
        )
        objfunc_i = ObjectiveFunction(inflow_i, twist_i, normalized_radius)
        if iteration_count == max_iteration:
            print("Scheme failed to converge")
            return [inflow_i, tip_twist_i]
    return [inflow_i, tip_twist_i]


def ExactInflow(normalized_radius: float) -> float:
    coefficient_thrust_given = 0.008
    return np.sqrt(coefficient_thrust_given / 2)


def DeltaCoefficientThrust(normalized_radius: float, step_size) -> float:
    """
    Using BEMT we find the discrete value for the inflow and twist at that
    radius value

    Note: we are using Ideal Twist

    Returns:
    discrete value
    """
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    inflow, tip_twist = FindInflowAndTipTwist(normalized_radius)
    delta_CT = (
        (solidity * coefficient_lift_alpha / 2)
        * (tip_twist - inflow)
        * normalized_radius
        * step_size
    )
    return delta_CT


def DeltaCoefficientThrustExact(normalized_radius):
    coefficient_thrust_given = 0.008
    return 2 * coefficient_thrust_given * normalized_radius


def CoefficientThrustBEMT(normalized_radius: float, step_size: float) -> float:
    r"""
    Finds the discrete sum up to the r value

    Returns:
    $\sum{\Delta{C_T}(r)}$
    """

    delta_CT_list = [
        DeltaCoefficientThrust(r, step_size)
        for r in np.arange(0.1, normalized_radius, step_size)
    ]
    return np.sum(delta_CT_list)


def CoefficientThrust_scuffed(final_C_T: float, normalized_radius: float) -> float:
    return (final_C_T / np.exp(1)) * np.exp(normalized_radius**2)


def CoefficientPowerExact(normalized_radius: float) -> float:
    inflow_ratio = ExactInflow(normalized_radius)
    coefficient_drag_0 = 0.008
    solidity = 0.1
    return (
        2 * inflow_ratio**3 * normalized_radius**2
        + (1 / 8) * solidity * coefficient_drag_0 * normalized_radius**4
    )


def DeltaCoefficientPowerBEMT(normalized_radius: float, step_size: float) -> float:
    inflow = FindInflowAndTipTwist(normalized_radius)[0]
    delta_CT = DeltaCoefficientThrust(normalized_radius, step_size)
    coefficient_drag_0 = 0.008
    solidity = 0.1
    return (
        inflow * delta_CT
        + 0.5 * solidity * coefficient_drag_0 * normalized_radius**3 * step_size
    )


def CoefficientPowerBEMT(normalized_radius: float, step_size: float) -> float:
    """
    Finds the Coefficient of Power at a given radius

    Returns:
    C_P
    """
    delta_CT = [
        DeltaCoefficientPowerBEMT(r, step_size)
        for r in np.arange(0.1, normalized_radius, step_size)
    ]
    return np.sum(delta_CT)


def DeltaCoefficientPowerDeltaRadiusExact(normalized_radius: float) -> float:
    """
    Finds the Exact derivative Coefficient of Power at a given radius

    Returns:
    dC_P/dr
    """
    coefficient_thrust_given = 0.008
    inflow = np.sqrt(coefficient_thrust_given / 2)
    dCTdr = DeltaCoefficientThrustExact(normalized_radius)
    coefficient_drag_0 = 0.008
    solidity = 0.1
    return inflow * dCTdr + 0.5 * solidity * coefficient_drag_0 * normalized_radius**3


def CoefficientLiftExact(normalized_radius: float) -> float:
    coefficient_thrust_given = 0.008
    solidity = 0.1
    return (4 * coefficient_thrust_given) / (solidity * normalized_radius)


def CoefficientLiftBEMT(normalized_radius: float) -> float:
    coefficient_lift_alpha = 2 * np.pi
    inflow, tip_twist = FindInflowAndTipTwist(normalized_radius)
    return (coefficient_lift_alpha / normalized_radius) * (tip_twist - inflow)


def CoefficientPowerInducedExact(given_coefficient_thrust: float) -> float:
    return given_coefficient_thrust**1.5 / 2**0.5


def CoefficientPowerInducedBEMT(normalized_radius: float, step_size: float) -> float:
    inflow_tip = FindInflowAndTipTwist(1)[0]
    n = 0
    inflow = inflow_tip * normalized_radius**n
    kappa = (2 * (n + 1) ** 0.5) / (3 * n + 2)
    delta_CP_i = [
        FindInflowAndTipTwist(r)[0] * DeltaCoefficientThrust(r, step_size)
        for r in np.arange(0.1, normalized_radius, step_size)
    ]
    # delta_CP_i = [4*FindInflowAndTipTwist(r)[0]**3 * r * step_size for r in np.arange(0.1,normalized_radius,step_size)]
    # return kappa * given_coefficient_thrust**1.5 / 2**0.5
    return np.sum(delta_CP_i)


def PlotProblem1():
    step_size = 1 / 100
    radius_values = np.arange(0.1, 1, step_size)
    inflow_values_bemt = [FindInflowAndTipTwist(radius)[0] for radius in radius_values]
    inflow_values_exact = [ExactInflow(r) for r in radius_values]

    plt.plot(radius_values, inflow_values_exact, label="Exact")
    plt.plot(radius_values, inflow_values_bemt, label="BEMT", linestyle=":")
    plt.title("Inflow Ratio")
    plt.legend()
    plt.xlabel(r"$r = \frac{y}{R}$")
    plt.ylabel(r"$\lambda$")
    plt.show()

    delta_CT_delta_radius_bemt = [
        DeltaCoefficientThrust(r, step_size) / step_size for r in radius_values
    ]
    delta_CT_delta_radius_exact = [
        DeltaCoefficientThrustExact(r) for r in radius_values
    ]
    plt.plot(radius_values, delta_CT_delta_radius_exact, label="Exact")
    plt.plot(radius_values, delta_CT_delta_radius_bemt, label="BEMT", linestyle=":")
    plt.title(r"$\frac{dC_T}{dx}$")
    plt.legend()
    plt.xlabel("r")
    plt.ylabel(r"$\frac{dC_T}{dx}$")
    plt.show()

    CT_bemt = [CoefficientThrustBEMT(r, step_size) for r in radius_values]
    # plt.plot(
    # radius_values,
    # CoefficientThrust_scuffed(CT_bemt[-1], radius_values),
    # label="Exact",
    # )
    plt.plot(radius_values, CT_bemt, label="BEMT", linestyle=":")
    plt.title("Coefficient of Thrust")
    plt.legend()
    plt.xlabel("r")
    plt.ylabel(r"$C_T$")
    plt.show()

    CP_bemt = [CoefficientPowerBEMT(r, step_size) for r in radius_values]
    plt.plot(radius_values, CoefficientPowerExact(radius_values), label="Exact")
    plt.plot(radius_values, CP_bemt, label="BEMT", linestyle=":")
    plt.title("Coefficient of Torque")
    plt.legend()
    plt.xlabel("r")
    plt.ylabel(r"$C_Q$")
    plt.show()

    delta_CP_delta_r = [
        DeltaCoefficientPowerBEMT(r, step_size) / step_size for r in radius_values
    ]
    plt.plot(
        radius_values,
        DeltaCoefficientPowerDeltaRadiusExact(radius_values),
        label="Exact",
    )
    plt.plot(radius_values, delta_CP_delta_r, label="BEMT", linestyle=":")
    plt.title(r"$\frac{dC_q}{dr}$")
    plt.legend()
    plt.xlabel("r")
    plt.ylabel(r"$\frac{dC_q}{dr}$")
    plt.show()

    plt.plot(radius_values, CoefficientLiftExact(radius_values), label="Exact")
    plt.plot(
        radius_values,
        [CoefficientLiftBEMT(r) for r in radius_values],
        label="BEMT",
        linestyle=":",
    )
    plt.title("Coefficient of Lift")
    plt.legend()
    plt.xlabel("r")
    plt.ylabel(r"$C_l$")
    plt.show()

    coefficient_thrust_values = np.linspace(0, CT_bemt[-1], len(radius_values))
    plt.plot(
        coefficient_thrust_values,
        CoefficientPowerInducedExact(coefficient_thrust_values),
        label="Exact",
    )
    plt.plot(
        coefficient_thrust_values,
        [CoefficientPowerInducedBEMT(r, step_size) for r in radius_values],
        label="BEMT",
        linestyle=":",
    )
    plt.title("Coefficient of Power Induced")
    plt.legend()
    plt.xlabel(r"$C_T$")
    plt.ylabel(r"$C_{P,i}$")
    # plt.ylim(-0.0,0.001)
    plt.show()
    return None


def SubPlotProblem1():
    step_size = 1 / 100
    radius_values = np.arange(0.1, 1, step_size)
    inflow_values_bemt = [FindInflowAndTipTwist(radius)[0] for radius in radius_values]
    inflow_values_exact = [ExactInflow(r) for r in radius_values]

    delta_CT_delta_radius_bemt = [
        DeltaCoefficientThrust(r, step_size) / step_size for r in radius_values
    ]
    delta_CT_delta_radius_exact = [
        DeltaCoefficientThrustExact(r) for r in radius_values
    ]

    CT_bemt = [CoefficientThrustBEMT(r, step_size) for r in radius_values]
    CP_bemt = [CoefficientPowerBEMT(r, step_size) for r in radius_values]

    delta_CP_delta_r = [
        DeltaCoefficientPowerBEMT(r, step_size) / step_size for r in radius_values
    ]
    coefficient_thrust_values = np.linspace(0, CT_bemt[-1], len(radius_values))

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 14), sharey=False)

    # Plot the first subplot for inflow ratio
    axes[0, 0].plot(radius_values, inflow_values_exact, label="Exact")
    axes[0, 0].plot(radius_values, inflow_values_bemt, label="BEMT", linestyle=":")
    axes[0, 0].set_title(r"$\lambda$")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel(r"$r = \frac{y}{R}$")
    axes[0, 0].set_ylabel("Inflow Ratio")

    # Plot the second subplot for dC_T/dx
    axes[1, 0].plot(radius_values, delta_CT_delta_radius_exact, label="Exact")
    axes[1, 0].plot(
        radius_values, delta_CT_delta_radius_bemt, label="BEMT", linestyle=":"
    )
    axes[1, 0].set_title(r"$\frac{dC_T}{dx}$")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("r")

    # Plot the third subplot for C_T
    axes[2, 0].plot(radius_values, CT_bemt, label="BEMT", linestyle=":")
    axes[2, 0].set_title("Coefficient of Thrust")
    axes[2, 0].legend()
    axes[2, 0].set_xlabel("r")

    # Plot the fourth subplot for C_Q
    axes[3, 0].plot(radius_values, CoefficientPowerExact(radius_values), label="Exact")
    axes[3, 0].plot(radius_values, CP_bemt, label="BEMT", linestyle=":")
    axes[3, 0].set_title("Coefficient of Torque")
    axes[3, 0].legend()
    axes[3, 0].set_xlabel("r")

    # Plot the fifth subplot for dC_q/dr
    axes[2, 1].plot(
        radius_values,
        DeltaCoefficientPowerDeltaRadiusExact(radius_values),
        label="Exact",
    )
    axes[2, 1].plot(radius_values, delta_CP_delta_r, label="BEMT", linestyle=":")
    axes[2, 1].set_title(r"$\frac{dC_q}{dr}$")
    axes[2, 1].legend()
    axes[2, 1].set_xlabel("r")

    # Plot the sixth subplot for C_l
    axes[0, 1].plot(
        radius_values,
        CoefficientLiftExact(radius_values),
        label="Exact",
    )
    axes[0, 1].plot(
        radius_values,
        [CoefficientLiftBEMT(r) for r in radius_values],
        label="BEMT",
        linestyle=":",
    )
    axes[0, 1].set_title("Coefficient of Lift")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("r")

    # Plot the seventh subplot for C_P,i
    axes[1, 1].plot(
        coefficient_thrust_values,
        CoefficientPowerInducedExact(coefficient_thrust_values),
        label="Exact",
    )
    axes[1, 1].plot(
        coefficient_thrust_values,
        [CoefficientPowerInducedBEMT(r, step_size) for r in radius_values],
        label="BEMT",
        linestyle=":",
    )
    axes[1, 1].set_title("Coefficient of Power Induced")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel(r"$C_T$")
    axes[1, 1].set_ylabel(r"$C_{P,i}$")

    # Adjust layout to prevent overlap between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()

    return None


if __name__ == "__main__":
    SubPlotProblem1()
