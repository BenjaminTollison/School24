import matplotlib.pyplot as plt
import numpy as np
from numba import njit, vectorize

solidity = 0.1
coefficient_thrust_hover = 0.008
inflow_hover_exact = lambda coefficient_thrust_hover: np.sqrt(
    coefficient_thrust_hover / 2
)


def IdealTwistTip(inflow_ratio_i: float) -> float:
    coefficient_lift_alpha = 2 * np.pi
    twist_tip = (4 * coefficient_thrust_hover) / (
        solidity * coefficient_lift_alpha
    ) + inflow_ratio_i
    return twist_tip


def InflowBEMTHover(tip_twist: float) -> float:
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    first_term = solidity * coefficient_lift_alpha / 16
    second_term = (
        np.sqrt(1 + abs((32 * tip_twist)) / (solidity * coefficient_lift_alpha)) - 1
    )
    return first_term * second_term


def DeltaCoefficientThrust(
    radius: float, radius_step_size: float, tip_twist: float, inflow_ratio: float
) -> float:
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    first_term = solidity * coefficient_lift_alpha / 2
    second_term = tip_twist - inflow_ratio
    third_term = radius * radius_step_size
    return first_term * second_term * third_term


def CoefficientThrustSum(
    radius_array: float, radius_step_size: float, tip_twist: float, inflow_ratio: float
) -> float:
    delta_C_T_array = np.zeros(shape=(radius_array).shape)
    for index in range(len(radius_array)):
        delta_C_T_array[index] = DeltaCoefficientThrust(
            radius_array[index], radius_step_size, tip_twist, inflow_ratio
        )
    return np.sum(delta_C_T_array)


def HoverBEMT(C_T_req: float) -> dict:
    number_of_divisions = 100
    delta_radius = 1 / number_of_divisions
    radius_values = np.arange(start=0.1, stop=1.0, step=delta_radius)
    solidity = 0.1
    coefficient_lift_alpha = np.pi
    intial_inflow = np.sqrt(1 / 2)
    intial_tip_twist = 4 * C_T_req / (solidity * coefficient_lift_alpha)
    tolerance = 1e-9
    solution_dict = {"radius": [], "inflow": [], "tip_twist": [], "C_T": []}
    for radius in radius_values:
        while (
            abs(
                C_T_req
                - CoefficientThrustSum(
                    radius_array=radius_values,
                    radius_step_size=delta_radius,
                    tip_twist=intial_tip_twist,
                    inflow_ratio=intial_inflow,
                )
            )
            >= tolerance
        ):
            numerical_C_T = CoefficientThrustSum(
                radius_array=radius_values,
                radius_step_size=delta_radius,
                tip_twist=intial_tip_twist,
                inflow_ratio=intial_inflow,
            )
            intial_tip_twist = (
                IdealTwistTip(intial_inflow)
                + (6 * (C_T_req - numerical_C_T)) / (solidity * coefficient_lift_alpha)
                + (3 * 2**0.5 / 4) * (np.sqrt(C_T_req) - np.sqrt(abs(numerical_C_T)))
            )
            intial_inflow = InflowBEMTHover(intial_tip_twist)
        final_C_T = CoefficientThrustSum(
            radius_array=radius_values,
            radius_step_size=delta_radius,
            tip_twist=intial_tip_twist,
            inflow_ratio=intial_inflow,
        )
        solution_dict["radius"].append(radius)
        solution_dict["C_T"].append(final_C_T)
        solution_dict["inflow"].append(intial_inflow)
        solution_dict["tip_twist"].append(intial_tip_twist)

    return solution_dict


if __name__ == "__main__":
    number_of_divisions = 100
    delta_radius = 1 / number_of_divisions
    radius_values = np.arange(start=0.1, stop=1, step=delta_radius)
    coefficient_thrust_hover = 0.008
    HoverBEMT(coefficient_thrust_hover)
    print("Check for errors")
