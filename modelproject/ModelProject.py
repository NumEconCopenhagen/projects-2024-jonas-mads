# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Setting model parameters for both the basic and extended Solow model
A_val, s_val, alpha_val, delta_val, n_val, g_val = 3, 0.2, 0.3, 0.4, 0.05, 0.05
x_init_val = 0.25
x_min_val, x_max_val = 0, 3

# Function to calculate the next period's capital stock
def f_function(A, s, alpha, delta, k):
    return A * s * k**alpha + (1 - delta) * k

# Function to calculate the next period's capital stock in the extended Solow model
def f_function_extended(A, s, alpha, delta, n, g, k):
    return A * s * k**alpha + (n + g + delta) * k

# Function to plot the 45-degree line and the f(k) function
def plot_diagonal(k_star=None,extended=0):
    # Generating a grid of values for the capital stock
    x_grid = np.linspace(x_min_val, x_max_val, 12000)

    # Creating a new figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(x_min_val, x_max_val)

    # Calculating values for the f(k) function for the two solow models
    if extended == 1:
        f_values = f_function_extended(A_val, s_val, alpha_val, delta_val, n_val, g_val, x_grid)
        label_f = r'$f(k) = sAk^{\alpha} + (n + g + \delta)k$' # Label for the f(k) function
    else:
        f_values = f_function(A_val, s_val, alpha_val, delta_val, x_grid)
        label_f = r'$f(k) = sAk^{\alpha} + (1 - \delta)k$' # Label for the f(k) function
        
    y_min, y_max = np.min(f_values), np.max(f_values)
    ax.set_ylim(y_min, y_max)
    
    # Plotting the f(k) function
    ax.plot(x_grid, f_values,  lw=2, alpha=0.8, label=label_f, color='red')

    # Plotting the 45-degree line
    ax.plot(x_grid, x_grid, 'b--', lw=1, alpha=0.7, label='$45^{\circ}$')

    # Plotting the steady-state capital point if provided
    if k_star:
        fps = (k_star,)
        ax.plot(fps, fps, 'yo', ms=10, alpha=0.8)
        ax.annotate(r'$k^*$',
                 xy=(k_star, k_star),
                 xycoords='data',
                 xytext=(-40, -60),
                 textcoords='offset points',
                 fontsize=14,
                 arrowprops=dict(arrowstyle="->", color='black'))

    # Adding legend and setting ticks and labels
    ax.legend(loc='upper left', frameon=False, fontsize=12)
    ax.set_xticks((0, 1, 2, 3))
    ax.set_yticks((0, 1, 2, 3))
    ax.set_xlabel('$k_t$', fontsize=12)
    ax.set_ylabel('$k_{t+1}$', fontsize=12)

    # Adding grid lines
    ax.grid(True, linestyle='--', alpha=0.5)
    # Formatting axis labels
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    # Displaying the plot
    plt.show()

def find_steady_state_capital(A, s, alpha, delta, k_guess=1.0 , tolerance=1e-6, max_iterations=10000):
    """
    Function to find the steady-state capital using iteration.

    Parameters:
        A (float): Total factor productivity.
        s (float): Savings rate.
        alpha (float): Capital share in output.
        delta (float): Depreciation rate.
        k_guess (float): Initial guess for steady-state capital. Default is 1.0.
        tolerance (float): Tolerance level for convergence. Default is 1e-6.
        max_iterations (int): Maximum number of iterations. Default is 10000.

    Returns:
        float: Steady-state capital.
    """
    for i in range(max_iterations):
        k_next = f_function(A, s, alpha, delta, k_guess)
        
        # Check for convergence
        if abs(k_next - k_guess) < tolerance:
            return k_next
        
        k_guess = k_next  # Update guess for the next iteration
        
    # If not converged within the maximum iterations, return None
    return None

# Function to plot the 45-degree line and the f(k) function
def plot_diagonal_params(A, s, alpha, delta, color, label, k_star=None):
    # Generating a grid of values for the capital stock
    x_grid = np.linspace(x_min_val, x_max_val, 12000)

    # Calculating values for the f(k) function
    f_values = f_function(A, s, alpha, delta, x_grid)
    #y_min, y_max = np.min(f_values), np.max(f_values)

    # Plotting the f(k) function
    plt.plot(x_grid, f_values,  lw=2, alpha=0.8, label=label, color=color)

    # Plotting the steady-state capital point if provided
    if k_star:
        plt.plot(k_star, k_star, 'o', ms=10, alpha=0.8, color=color)

def plot_scenarios(parameters):
    # Creating a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    x_grid = np.linspace(x_min_val, x_max_val, 12000)

    k_star_values = []

    # Plotting for each scenario
    for params in parameters:
        k_star_val = find_steady_state_capital(params['A'], params['s'], params['alpha'], params['delta'])
        k_star_values.append(k_star_val)
        plot_diagonal_params(params['A'], params['s'], params['alpha'], params['delta'], params['color'], params['label'], k_star_val)

    # plots the 45 degree line
    plt.plot(x_grid, x_grid, 'b--', lw=1, alpha=0.7, label='$45^{\circ}$')

    # Adding legend and setting ticks and labels
    plt.legend(loc='upper left', frameon=False, fontsize=12)
    plt.xlabel('$k_t$', fontsize=12)
    plt.ylabel('$k_{t+1}$', fontsize=12)
    plt.xticks(np.arange(0, 3.1, 0.5))
    plt.yticks(np.arange(0, 3.1, 0.5))

    # Adding grid lines
    plt.grid(True, linestyle='--', alpha=0.5)

    # Displaying the plot
    plt.show()

    return k_star_values

# Function to find the steady-state capital per effective worker (k*)
def find_steady_state_extended(A, s, alpha, delta, n, g):
    k_star = ((s * A) / (n + g + delta))**(1 / (1 - alpha))
    return k_star
