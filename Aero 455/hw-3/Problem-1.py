import matplotlib.pyplot as plt
import numpy as np

solidity = 0.1
coefficient_thrust_hover = 0.008
inflow_hover = np.sqrt(coefficient_thrust_hover / 2)
print(inflow_hover)


def IdealTwist(normalized_radius: float) -> float:
    twist_tip = (4 * coefficient_thrust_hover) / (solidity * np.pi) + np.sqrt(
        coefficient_thrust_hover / 2
    )
    return twist_tip / normalized_radius


def InflowBEMTHover(normalized_radius: float) -> float:
    solidity = 0.1
    coefficient_lift_alpha = 2 * np.pi
    first_term = np.sqrt(
        solidity * coefficient_lift_alpha / 16
        + solidity * IdealTwist(normalized_radius) * normalized_radius / 8
    )
    second_term = solidity * coefficient_lift_alpha / 16
    return first_term - second_term


radius_values = np.linspace(0.1, 1, 100)
inflow_hover_bemt_values = [InflowBEMTHover(r) for r in radius_values]
inflow_hover_exact_values = np.full(len(radius_values), inflow_hover)

if __name__ == "__main__":

    plt.plot(radius_values, inflow_hover_exact_values, label="exact")
    plt.plot(radius_values, inflow_hover_bemt_values, label="BEMT", linestyle="--")
    plt.legend()
    plt.title("Inflow Ratio")
    plt.xlabel(r"$ r = \frac{y}{R}$")
    plt.ylabel(r"$\lambda$")
    plt.show()
### I hate this, its going to have to be re-written.
