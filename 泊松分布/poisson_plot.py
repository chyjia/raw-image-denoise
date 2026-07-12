import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def poisson_pmf(k, lam):
    """Return P(X = k) for a Poisson distribution with mean lam."""
    return (lam**k) * math.exp(-lam) / math.factorial(k)


def parse_lambdas(raw_input):
    values = []
    for item in raw_input.split(","):
        item = item.strip().lstrip("\ufeff")
        if not item:
            continue

        lam = float(item)
        if lam <= 0:
            raise ValueError("lambda values must be greater than 0")
        values.append(lam)

    if not values:
        raise ValueError("please enter at least one lambda value")

    return values


def main():
    raw_input = input("Enter Poisson mean values separated by commas, e.g. 2,5,10: ")
    lambdas = parse_lambdas(raw_input)

    plt.figure(figsize=(9, 5))

    for lam in lambdas:
        max_k = int(lam + 4 * math.sqrt(lam)) + 10
        k_values = list(range(max_k + 1))
        probabilities = [poisson_pmf(k, lam) for k in k_values]

        plt.plot(k_values, probabilities, marker="o", linewidth=1.5, label=f"lambda = {lam:g}")

    plt.title("Poisson Distribution")
    plt.xlabel("k")
    plt.ylabel("P(X = k)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = "poisson_distribution.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
