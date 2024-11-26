import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from Problem2 import (
    TipLossFactor,
    TipLossFunction,
)
from Problem4 import CoefficientDrag
from GpuProblem5 import PlotGpuResults, RunGPUFunctions, TroubleShootingPlots

number_of_mistakes_i_am_willing_to_get_that_grade_otherwise_i_will_q_drop_off_of_rudder_tower_jk_this_project_is_actually_kinda_fun_after_you_consume_enough_caffiene = (
    103
)
delta_radius = (
    1
    / number_of_mistakes_i_am_willing_to_get_that_grade_otherwise_i_will_q_drop_off_of_rudder_tower_jk_this_project_is_actually_kinda_fun_after_you_consume_enough_caffiene
)
starting_radius = 0.1


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
    normalized_radius,
    twist,
    number_of_blades: int,
    taper,
):
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
    while np.all(abs(objfunc_i) >= tolerance):
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

    return inflow_i, tip_function_i


def DeltaCoefficientThrust(
    normalized_radius: float,
    twist: float,
    number_of_blades: int,
    taper: float,
) -> float:
    twist = LinearTwist(normalized_radius, twist, number_of_blades, taper)
    inflow, F = InflowBEMT(normalized_radius, twist, number_of_blades, taper)
    return 4 * F * inflow**2 * normalized_radius * delta_radius


def DeltaCoefficientPowerInduced(
    normalized_radius: float,
    twist: float,
    number_of_blades: int,
    taper: float,
) -> float:
    twist = LinearTwist(normalized_radius, twist, number_of_blades, taper)
    inflow = InflowBEMT(normalized_radius, twist, number_of_blades, taper)[0]
    delta_CT = DeltaCoefficientThrust(
        normalized_radius,
        twist,
        number_of_blades,
        taper,
    )
    return inflow * delta_CT


def CoefficientLift(
    normalized_radius: float,
    twist: float,
    number_of_blades: int,
    taper: float,
) -> float:
    coefficient_lift_alpha = 2 * np.pi
    twist = LinearTwist(normalized_radius, twist, number_of_blades, taper)
    inflow, F = InflowBEMT(normalized_radius, twist, number_of_blades, taper)
    solidity = Solidity(normalized_radius, number_of_blades, taper)
    return (8 / (solidity * normalized_radius)) * F * inflow**2


def CoefficientPowerInduced(
    normalized_radius: float,
    twist: float,
    number_of_blades: int,
    taper: float,
) -> float:
    delta_CP_list = [
        DeltaCoefficientPowerInduced(r, twist, number_of_blades, taper)
        for r in np.arange(starting_radius, normalized_radius, delta_radius)
    ]
    return np.sum(delta_CP_list)


def CoefficientThrust(
    normalized_radius: float,
    twist: float,
    number_of_blades: int,
    taper: float,
) -> float:
    delta_CT_list = [
        DeltaCoefficientThrust(r, twist, number_of_blades, taper)
        for r in np.arange(starting_radius, normalized_radius, delta_radius)
    ]
    return np.sum(delta_CT_list)


def CoefficientPowerIdeal(
    normalized_radius: float,
    twist: float,
    number_of_blades: int,
    taper: float,
) -> float:
    return (
        CoefficientThrust(normalized_radius, twist, number_of_blades, taper) ** 1.5
        / 2**0.5
    )
    # return 0.008**1.5 / 2**0.5


def CoefficientPower(
    normalized_radius: float,
    twist: float,
    number_of_blades: int,
    taper: float,
) -> float:
    delta_CP_list = [
        InflowBEMT(r, twist, number_of_blades, taper)[0]
        * DeltaCoefficientThrust(r, twist, number_of_blades, taper)
        + 0.5
        * Solidity(r, number_of_blades, taper)
        * CoefficientDrag(r, number_of_blades, twist)
        * r**3
        * delta_radius
        for r in np.arange(starting_radius, normalized_radius, delta_radius)
    ]
    return np.sum(delta_CP_list)


def FigureOfMerit(
    normalized_radius: float,
    twist: float,
    number_of_blades: int,
    taper: float,
) -> float:
    CP_ideal = CoefficientPowerIdeal(normalized_radius, twist, number_of_blades, taper)
    CP = CoefficientPower(normalized_radius, twist, number_of_blades, taper)
    return CP_ideal / CP


def PlotProblem5():
    fig, axis = plt.subplots(ncols=2, nrows=2, figsize=(8, 10))

    radius_values = np.arange(starting_radius, 1, delta_radius)
    fixed_twist_rate = np.deg2rad(-15)
    taper_ratio_values = [1, 2, 3]
    # Plot the first subplot
    for taper_ratio in taper_ratio_values:
        axis[0, 0].plot(
            radius_values,
            [InflowBEMT(r, fixed_twist_rate, 2, taper_ratio)[0] for r in radius_values],
            label=r"$\pi_{taper} = $" + f"{round(taper_ratio,0)}",
            linestyle="-",
        )
        axis[0, 1].plot(
            radius_values,
            [
                DeltaCoefficientThrust(r, fixed_twist_rate, 2, taper_ratio)
                / delta_radius
                for r in radius_values
            ],
            label=r"$\pi_{taper} = $" + f"{round(taper_ratio,0)}",
            linestyle="-",
        )
        axis[1, 0].plot(
            radius_values,
            [
                DeltaCoefficientPowerInduced(r, fixed_twist_rate, 2, taper_ratio)
                / delta_radius
                for r in radius_values
            ],
            label=r"$\pi_{taper} = $" + f"{round(taper_ratio,0)}",
            linestyle="-",
        )
        axis[1, 1].plot(
            radius_values,
            [
                CoefficientLift(r, fixed_twist_rate, 2, taper_ratio)
                for r in radius_values
            ],
            label=r"$\pi_{taper} = $" + f"{round(taper_ratio,0)}",
            linestyle="-",
        )

    # First plot formatting
    axis[0, 0].set_title("Inflow Ratio")
    axis[0, 0].legend()
    axis[0, 0].set_xlabel(r"$r = \frac{y}{R}$")
    axis[0, 0].set_ylabel(r"$\lambda$")
    # Second plot formatting
    axis[0, 1].set_title("Change in Coefficient in Thrust")
    axis[0, 1].legend()
    axis[0, 1].set_xlabel(r"$r$")
    axis[0, 1].set_ylabel(r"$\frac{dC_T}{dr}$")
    # Third plot formatting
    axis[1, 0].set_title("Change in Coefficient of Induced Power")
    axis[1, 0].legend()
    axis[1, 0].set_xlabel(r"$r$")
    axis[1, 0].set_ylabel(r"$\frac{dC_{Pi}}{dr}$")
    # Fourth plot formatting
    axis[1, 1].set_title("Change in Coefficient of Lift")
    axis[1, 1].legend()
    axis[1, 1].set_xlabel(r"$r$")
    axis[1, 1].set_ylabel(r"$C_l$")

    # Adjust layout to prevent overlap between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()

    return None


def RunCpuFunctions(cpu_mesh_size=100):
    twist_rate = np.deg2rad(np.linspace(-15, 15, cpu_mesh_size))
    taper = np.linspace(1, 6, cpu_mesh_size)

    X, Y = np.meshgrid(twist_rate, taper)
    with tqdm(total=X.shape[0] * X.shape[1], desc="Computing FM values") as pbar:
        Z = np.zeros_like(X)  # Ensure Z has the same shape as X and Y
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = FigureOfMerit(1, X[i, j], 2, Y[i, j])
                pbar.update(1)
    return Z


def CompareRuntimes(mesh_size=100):
    cpu_start = time()
    RunCpuFunctions(mesh_size)
    cpu_runtime = time() - cpu_start
    gpu_start = time()
    RunGPUFunctions(mesh_size)
    gpu_runtime = time() - gpu_start
    print(f"For a mesh size of {mesh_size}")
    print(f"The CPU had a runtime of {cpu_runtime:.6} seconds")
    print(f"The GPU had a runtime of {gpu_runtime:.6} seconds")
    print("===========================================================")
    print(
        f"|           The GPU runs {cpu_runtime/gpu_runtime:.8} times faster           |"
    )
    print("===========================================================")


def FigureOfMerit3DPlot():
    try:

        PlotGpuResults()
        TroubleShootingPlots()

        mesh_sizes_to_compare = [10, 25, 50, 100]
        print("Comparing the computation time of each method")
        for mesh in mesh_sizes_to_compare:
            CompareRuntimes(mesh)
    except (
        ModuleNotFoundError,
        RuntimeError,
        OSError,
        Exception,
        NotImplementedError,
    ) as e:
        print(f"Error occurred with CuPy: {e}. Falling back to NumPy...")
        # Add fallback logic with NumPy here
        X, Y, Z = RunCpuFunctions()

        # Create a figure and an axis for the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the data as a scatter plot
        surf = ax.plot_surface(np.degrees(X), Y, Z, cmap="viridis")

        # Set labels for the axes
        ax.set_xlabel("Twist Rate")
        ax.set_ylabel("Taper")
        ax.set_zlabel("FM")

        # Rotate the plot to see a specific plane
        # For example, rotate around the x-axis by 30 degrees and around the z-axis by 45 degrees
        ax.view_init(elev=20, azim=-130)

        # Add a color bar to show the mapping of colors to values
        fig.colorbar(surf, shrink=0.5)

        # Display the plot
        plt.show()


if __name__ == "__main__":
    PlotProblem5()
    FigureOfMerit3DPlot()
