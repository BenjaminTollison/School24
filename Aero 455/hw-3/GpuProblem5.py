import numpy as np
import cupy as cp
from tqdm import tqdm


def SolidityVectorized(
    radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
):
    root_chord = 0.1 * np.pi / number_of_blades
    return (number_of_blades / (xp.pi * radius_vector)) * (
        root_chord - root_chord * (1 - 1 / taper_vector) * radius_vector
    )


def LinearTwistVectorize(
    radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
):
    solidity = SolidityVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector
    )
    coefficient_lift_alpha = 2 * xp.pi
    coefficient_thrust_required = 0.008
    twist_0 = (
        (6 * coefficient_thrust_required) / (solidity * coefficient_lift_alpha)
        - 0.75 * twist_rate_vector
        + 1.5 * xp.sqrt(coefficient_thrust_required / 2)
    )
    return xp.add(twist_0, xp.multiply(twist_rate_vector, radius_vector))


def TipLossFactorVectorized(radius_vector, number_of_blades, inflow_ratio, xp=np):
    return (number_of_blades / 2) * ((1 - radius_vector) / inflow_ratio)


def TipLossFunctionVectorized(tip_loss_factor, xp=np):
    return (2 / xp.pi) * xp.arccos(xp.exp(-tip_loss_factor))


def InflowWithTipLossVectorized(
    radius_vector,
    twist_rate_vector,
    number_of_blades,
    tip_loss,
    taper_vector,
    xp=np,
):
    solidity = SolidityVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector
    )
    coefficient_lift_alpha = 2 * xp.pi
    first_term = (solidity * coefficient_lift_alpha) / (16 * tip_loss)
    second_term = (
        xp.sqrt(
            1
            + ((32 * tip_loss) / (solidity * coefficient_lift_alpha))
            * twist_rate_vector
            * radius_vector
        )
        - 1
    )
    return first_term * second_term


def ObjectiveInflowFunctionVectorized(
    radius_vector,
    inflow_ratio,
    tip_loss,
    twist_rate_vector,
    number_of_blades,
    taper_vector,
    xp=np,
):
    solidity = SolidityVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector
    )
    coefficient_lift_alpha = 2 * xp.pi

    return (
        inflow_ratio**2
        + ((solidity * coefficient_lift_alpha) / (8 * tip_loss)) * inflow_ratio
        - ((solidity * coefficient_lift_alpha) / (8 * tip_loss))
        * twist_rate_vector
        * radius_vector
    )


def InflowBEMTVectorized(
    radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
):
    """
    ## Finds the inflow at given span of the blade as a function of the radius, number of blades, linear twist, and taper

    Returns:
    Inflow Ratio
    """
    twist_i = LinearTwistVectorize(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
    )
    inflow_i = InflowWithTipLossVectorized(
        radius_vector, number_of_blades, 1, twist_i, taper_vector, xp
    )
    tip_factor_i = TipLossFactorVectorized(
        radius_vector, number_of_blades, inflow_i, xp
    )
    tip_function_i = TipLossFunctionVectorized(tip_factor_i, xp)
    iteration_count = 0
    iteration_max = 50
    tolerance = 1e-4
    objfunc_i = ObjectiveInflowFunctionVectorized(
        radius_vector,
        inflow_i,
        tip_function_i,
        twist_i,
        number_of_blades,
        taper_vector,
        xp,
    )
    while np.all(abs(objfunc_i) >= tolerance):
        iteration_count += 1
        inflow_i = InflowWithTipLossVectorized(
            radius_vector,
            number_of_blades,
            tip_function_i,
            twist_i,
            taper_vector,
            xp,
        )
        tip_factor_i = TipLossFactorVectorized(
            radius_vector, number_of_blades, inflow_i, xp
        )
        tip_function_i = TipLossFunctionVectorized(tip_factor_i, xp)
        objfunc_i = ObjectiveInflowFunctionVectorized(
            radius_vector,
            inflow_i,
            tip_function_i,
            twist_i,
            number_of_blades,
            taper_vector,
            xp,
        )
        if iteration_count == iteration_max:
            print("Scheme didn't converge")
            print(
                [
                    inflow_i.shape,
                    tip_function_i.shape,
                    tip_factor_i.shape,
                    xp.linalg.norm(objfunc_i[0], 2),
                ]
            )
            return inflow_i, tip_function_i

    return inflow_i, tip_function_i


def CoefficientDragVectorized(
    radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
):
    twist = LinearTwistVectorize(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
    )
    inflow = InflowBEMTVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
    )[0]
    alpha = xp.subtract(twist, xp.divide(inflow, radius_vector))
    return 0.011 - 0.025 * alpha + 0.65 * alpha**2


def DeltaCoefficientThrustVectorized(
    radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
):
    (
        inflow,
        F,
    ) = InflowBEMTVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
    )
    return xp.multiply(
        4,
        xp.multiply(
            F,
            xp.multiply(
                xp.multiply(inflow, inflow),
                xp.multiply(
                    radius_vector, xp.subtract(radius_vector[1], radius_vector[0])
                ),
            ),
        ),
    )


def CoefficientThrustVectorized(
    radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
):
    solidity = SolidityVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
    )
    coefficient_lift_alpha = 2 * xp.pi
    linear_twist = LinearTwistVectorize(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
    )
    inflow = InflowBEMTVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
    )[0]
    return (solidity * coefficient_lift_alpha / 4) * (solidity - inflow)


def CoefficientPowerIdealVectorized(
    radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
):
    return xp.divide(
        xp.abs(
            CoefficientThrustVectorized(
                radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
            )
        )
        ** 1.5,
        2**0.5,
    )


def CoefficientPowerVectorized(
    radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
):
    delta_CP_list = InflowBEMTVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector
    )[0] * DeltaCoefficientThrustVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector
    ) + 0.5 * SolidityVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
    ) * CoefficientDragVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
    ) * radius_vector**3 * xp.subtract(
        radius_vector[1], radius_vector[0]
    )
    return delta_CP_list


def FigureOfMeritVectorized(
    radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
):
    return xp.divide(
        CoefficientPowerIdealVectorized(
            radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
        ),
        CoefficientPowerVectorized(
            radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
        ),
    )


def RunGPUFunctions(gpu_mesh_size=1000, func=FigureOfMeritVectorized):
    import cupy as cp
    from time import time

    # Free all memory blocks in CuPy's default memory pool
    cp.get_default_memory_pool().free_all_blocks()

    # Optionally, clear pinned memory pool as well
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # Get the number of available devices
    num_devices = cp.cuda.runtime.getDeviceCount()
    print(f"Number of GPUs available: {num_devices}")

    # Print details for each device
    for i in range(num_devices):
        device_props = cp.cuda.runtime.getDeviceProperties(i)
        print(f"Device {i}: {device_props['name']}")

    starting_radius = 0.2
    twist_rate = cp.deg2rad(cp.linspace(-5, 10, gpu_mesh_size))
    taper = cp.linspace(1, 6, gpu_mesh_size)
    print("CuPy detected. Running on GPU...")
    # Compute the meshgrid on the GPU
    twist_rate_vector, taper_rate_vector = cp.meshgrid(twist_rate, taper)
    # Compute Z values (parallelized on GPU)
    r_vector = cp.linspace(starting_radius, 1, gpu_mesh_size)
    start_time = time()
    if func == InflowBEMTVectorized:
        resultant = InflowBEMTVectorized(
            r_vector, twist_rate_vector, 2, taper_rate_vector, cp
        )[0]
    else:
        resultant = func(r_vector, twist_rate_vector, 2, taper_rate_vector, cp)
    end_time = time()
    # print(f"Result: {FM.shape}")
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    return twist_rate_vector, taper_rate_vector, resultant


def PlotGpuResults(gpu_mesh_size=5000, func=FigureOfMeritVectorized):
    import cupy as cp
    import matplotlib.pyplot as plt

    X, Y, Z = RunGPUFunctions(gpu_mesh_size, func)
    # Convert results back to NumPy for plotting
    X = X.get()
    Y = Y.get()
    Z = Z.get()
    # print("Z shape:", Z.shape)
    # print("Z dtype:", Z.dtype)
    # print("Z size (bytes):", Z.nbytes)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the data as a scatter plot
    surf = ax.plot_surface(np.rad2deg(X), Y, Z, cmap="viridis")

    # Set labels for the axes
    ax.set_xlabel("Twist Rate")
    ax.set_ylabel("Taper")
    ax.set_zlabel("FM")

    # Rotate the plot to see a specific plane
    # For example, rotate around the x-axis by 30 degrees and around the z-axis by 45 degrees
    ax.view_init(elev=20, azim=-130)

    # Add a color bar to show the mapping of colors to values
    fig.colorbar(surf, shrink=0.5)
    plt.title(f"{func}")

    # Display the plot
    plt.show()


def TroubleShootingPlots():
    PlotGpuResults(500, InflowBEMTVectorized)
    PlotGpuResults(500, CoefficientDragVectorized)
    PlotGpuResults(500, CoefficientThrustVectorized)
    PlotGpuResults(500, CoefficientPowerIdealVectorized)
    PlotGpuResults(500, CoefficientPowerVectorized)
    # print(RunGPUFunctions(10, FigureOfMeritVectorized)[2])
    PlotGpuResults(500, FigureOfMeritVectorized)


if __name__ == "__main__":
    # Get memory pool information
    mempool = cp.cuda.MemoryPool()
    used_bytes = mempool.used_bytes()
    total_bytes = cp.cuda.Device().mem_info[1]
    free_bytes = total_bytes - used_bytes

    print(f"Total Memory: {total_bytes / (1024 ** 3):.2f} GB")
    print(f"Free Memory: {free_bytes / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {used_bytes / (1024 ** 3):.2f} GB")
    # RunGPUFunctions()
    # PlotGpuResults(5000, FigureOfMeritVectorized)
    TroubleShootingPlots()
