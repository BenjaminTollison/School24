import numpy as np


def InflowRatio(normalized_radius, inflow_i, C_T_i, C_T_req, step_size):
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    first_term = solidity * coefficient_lift_alpha / 16
    second_term = (
        np.sqrt(
            1
            + (32 / (solidity * coefficient_lift_alpha))
            * Twist(normalized_radius, inflow_i, C_T_req)
            * normalized_radius
        )
        - 1
    )
    return first_term * second_term


def Twist(normalized_radius, inflow_i, C_T_req):
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    tip_twist = 4 * C_T_req / (solidity * coefficient_lift_alpha) + inflow_i
    return tip_twist / normalized_radius


def DeltaCoefficientThrust(normalized_radius, inflow_i, C_T_i, C_T_req, step_size):
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    inflow_ratio = InflowRatio(normalized_radius, inflow_i, C_T_i, C_T_req, step_size)
    twist = Twist(normalized_radius, inflow_i, C_T_req)
    return (
        (solidity * coefficient_lift_alpha / 2)
        * (twist * normalized_radius**2 - inflow_ratio * normalized_radius)
        * step_size
    )


def CoefficientThrustBEMT(C_T_req, inflow_i, C_T_i, step_size):
    radius_values = np.arange(start=0.1, stop=1.0 + step_size, step=step_size)
    total_thrust = np.sum(
        [
            DeltaCoefficientThrust(r, inflow_i, C_T_i, C_T_req, step_size)
            for r in radius_values
        ]
    )
    return total_thrust


C_T_req = 0.008


def BEMT(C_T_reqired):
    input = {
        "normalized_radius": 0.5,
        "inflow_i": np.sqrt(0.001 / 2),
        "C_T_i": C_T_reqired + 0.001,
        "C_T_req": C_T_reqired,
        "step_size": 1 / 100,
    }
    C_T_numerical = CoefficientThrustBEMT(
        C_T_req=input["C_T_req"],
        inflow_i=input["inflow_i"],
        C_T_i=input["C_T_i"],
        step_size=input["step_size"],
    )
    tolerance = 1e-9
    iteration_max = 100
    iteration_count = 0
    radius_values = np.arange(
        start=0.1, stop=1.0 + input["step_size"], step=input["step_size"]
    )
    for r in radius_values:
        while (
            abs(C_T_req - C_T_numerical) >= tolerance
            or iteration_count <= iteration_max
        ):
            input = {
                "normalized_radius": r,
                "inflow_i": input["inflow_i"],
                "C_T_i": input["C_T_i"],
                "C_T_req": input["C_T_req"],
                "step_size": 1 / 100,
            }
            # twist_i = Twist(r, input['inflow_i'], input['C_T_req']) + (3/2)*input['inflow_i']
            input["inflow_i"] = InflowRatio(
                r,
                input["inflow_i"],
                input["C_T_i"],
                input["C_T_req"],
                input["step_size"],
            )

            C_T_numerical = CoefficientThrustBEMT(
                C_T_req=input["C_T_req"],
                inflow_i=input["inflow_i"],
                C_T_i=input["C_T_i"],
                step_size=input["step_size"],
            )
            input["C_T_i"] = C_T_numerical
        print(input["inflow_i"])


BEMT(C_T_req)
