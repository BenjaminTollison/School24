import numpy as np

try:
    import cupy as cp

    starting_radius = 0.1
    gpu_mesh_size = 100
    gpu_delta_radius = 1 / gpu_mesh_size
    twist_rate = cp.deg2rad(cp.linspace(-15, 15, gpu_mesh_size))
    taper = cp.linspace(1, 6, gpu_mesh_size)
    print("CuPy detected. Running on GPU...")
    # Compute the meshgrid on the GPU
    twist_rate_vector, taper_rate_vector = cp.meshgrid(twist_rate, taper)
    # Compute Z values (parallelized on GPU)
    r_vector = cp.linspace(starting_radius, 1, gpu_mesh_size)

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
        iteration_max = 30
        tolerance = 1e-6
        objfunc_i = ObjectiveInflowFunctionVectorized(
            radius_vector,
            inflow_i,
            tip_function_i,
            twist_i,
            number_of_blades,
            taper,
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
                taper,
                xp,
            )
            if iteration_count == iteration_max:
                print("Scheme didn't converge")
                print([inflow_i, tip_function_i, tip_factor_i, abs(objfunc_i)])
                return inflow_i

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
        alpha = cp.subtract(twist, xp.divide(inflow, radius_vector))
        return 0.011 - 0.025 * alpha + 0.65 * alpha**2

    def DeltaCoefficientThrustVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
    ):
        twist = LinearTwistVectorize(
            radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
        )
        inflow, F = InflowBEMTVectorized(
            radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
        )
        return (
            4
            * F
            * inflow**2
            * radius_vector
            * xp.subtract(radius_vector[1], radius_vector[0])
        )

    def CoefficientThrustVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
    ):

        delta_CT_list = DeltaCoefficientThrustVectorized(
            radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp
        )
        return xp.sum(delta_CT_list)

    test_dct = DeltaCoefficientThrustVectorized(
        r_vector, twist_rate_vector, 2, taper_rate_vector, cp
    )
    test_ct = CoefficientThrustVectorized(
        r_vector, twist_rate_vector, 2, taper_rate_vector, cp
    )
    print(test_dct.shape)

    def CoefficientPowerIdealVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
    ):
        return (
            CoefficientThrustVectorized(
                radius_vector, twist_rate_vector, number_of_blades, taper_vector
            )
            ** 1.5
            / 2**0.5
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
        return xp.sum(delta_CP_list)

    def FigureOfMeritVectorized(
        radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
    ):

        Cp_ideal = CoefficientPowerIdealVectorized(
            radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
        )
        Cp = CoefficientPowerVectorized(
            radius_vector, twist_rate_vector, number_of_blades, taper_vector, xp=np
        )
        return xp.divide(Cp_ideal, Cp)

    Cp_ideal = CoefficientPowerIdealVectorized(
        r_vector, twist_rate_vector, 2, taper_rate_vector, cp
    )
    print(Cp_ideal.shape)
    print(
        FigureOfMeritVectorized(
            r_vector, twist_rate_vector, 2, taper_rate_vector, cp
        ).shape
    )

except (ModuleNotFoundError, RuntimeError, OSError, Exception) as e:
    print(f"Error occurred with CuPy: {e}. Falling back to NumPy...")
