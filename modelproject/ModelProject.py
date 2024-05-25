# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ipywidgets import interactive, FloatSlider

# Setting model parameters for both the basic and extended Solow model
A_val, s_val, alpha_val, delta_val, n_val, g_val = 3, 0.2, 0.3, 0.4, 0.05, 0.05
x_init_val = 0.25
x_min_val, x_max_val = 0, 3

# Function to calculate the next period's capital stock
def f_function(A, s, alpha, delta, k):
    return A * s * k**alpha + (1 - delta) * k

# Function to calculate the next period's capital stock for the extended solow model
def f_function_extended(A, s, alpha, delta, n, g, k):
    return A * s * k**alpha + (1 - (delta + n + g)) * k

def find_steady_state_capital(A, s, alpha, delta, k_guess=1.0 , tolerance=1e-6, max_iterations=10000):
    for i in range(max_iterations):
        k_next = f_function(A, s, alpha, delta, k_guess)
        
        # Check for convergence
        if abs(k_next - k_guess) < tolerance:
            return k_next
        
        k_guess = k_next  # Update guess for the next iteration
        
    # If not converged within the maximum iterations, return None
    return None

def find_steady_state_extended(A, s, alpha, delta, n, g, k_guess=1.0, tolerance=1e-6, max_iterations=10000):
    for i in range(max_iterations):
        k_next = f_function_extended(A, s, alpha, delta, n, g, k_guess)
        
        # Check for convergence
        if abs(k_next - k_guess) < tolerance:
            return k_next
        
        k_guess = k_next  # Update guess for the next iteration
        
    # If not converged within the maximum iterations, return None
    return None


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


def interactive_plot(A, s, alpha, delta):
    # Calculate the steady state to determine an appropriate range for x_grid
    k_star = find_steady_state_capital(A, s, alpha, delta)
    max_val = max(k_star * 1.1, 3)  # Ensure there is always a bit more space than k_star and at least up to 3

    # Set up the grid of capital values dynamically based on calculated k_star
    x_grid = np.linspace(0, max_val, 12000)

    # Calculate the function values for the dynamically adjusted x_grid
    f_values = f_function(A, s, alpha, delta, x_grid)
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_grid, f_values, 'r', label=r'$f(k) = sAk^{\alpha} + (1 - \delta)k$')
    ax.plot(x_grid, x_grid, 'b--', label='$45^{\circ}$')
    
    # Update axis limits based on the new dynamic range of f_values and k_star
    max_plot_val = max(np.max(f_values), k_star) * 1.1
    ax.set_xlim(0, max_plot_val)
    ax.set_ylim(0, max_plot_val)
    
    # Mark steady state if it is within the plot bounds
    if k_star < max_plot_val:
        ax.plot(k_star, k_star, 'yo', ms=10, label=f'Steady state at k* = {k_star:.2f}')
    
    # Customize plot
    ax.set_title('Solow Model Interactive Plot')
    ax.set_xlabel('$k_t$', fontsize=12)
    ax.set_ylabel('$k_{t+1}$', fontsize=12)
    ax.legend()
    ax.grid(True)


# Create interactive widgets
interactive_plot_widget = interactive(interactive_plot,
                                      A=FloatSlider(value=3, min=1, max=5, step=0.1, description='A'),
                                      s=FloatSlider(value=0.2, min=0.1, max=0.4, step=0.05, description='s'),
                                      alpha=FloatSlider(value=0.3, min=0.1, max=0.9, step=0.05, description='alpha'),
                                      delta=FloatSlider(value=0.4, min=0.1, max=0.6, step=0.05, description='delta'))

    

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

# Interactive plot function for the extended Solow model
def interactive_plot_extended(A, s, alpha, delta, n, g):
    k_star = find_steady_state_extended(A, s, alpha, delta, n, g)
    max_val = max(k_star * 1.1, 3)  # Ensure there is always a bit more space than k_star and at least up to 3
    x_grid = np.linspace(0, max_val, 12000)
    f_values = f_function_extended(A, s, alpha, delta, n, g, x_grid)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_grid, f_values, 'r', label=r'$f(k) = sAk^{\alpha} + (1 - (\delta + n + g))k$')
    ax.plot(x_grid, x_grid, 'b--', label='$45^{\circ}$')
    max_plot_val = max(np.max(f_values), k_star) * 1.1
    ax.set_xlim(0, max_plot_val)
    ax.set_ylim(0, max_plot_val)
    if k_star < max_plot_val:
        ax.plot(k_star, k_star, 'yo', ms=10, label=f'Steady state at k* = {k_star:.2f}')
    ax.set_title('Extended Solow Model Interactive Plot')
    ax.set_xlabel('$k_t$', fontsize=12)
    ax.set_ylabel('$k_{t+1}$', fontsize=12)
    ax.legend()
    ax.grid(True)

# Create interactive widgets
interactive_plot_extended_widget = interactive(interactive_plot_extended,
                                               A=FloatSlider(value=3, min=1, max=5, step=0.1, description='A'),
                                               s=FloatSlider(value=0.2, min=0.1, max=0.4, step=0.05, description='s'),
                                               alpha=FloatSlider(value=0.3, min=0.1, max=0.9, step=0.05, description='alpha'),
                                               delta=FloatSlider(value=0.4, min=0.1, max=0.6, step=0.05, description='delta'),
                                               n=FloatSlider(value=0.05, min=0, max=0.1, step=0.01, description='n'),
                                               g=FloatSlider(value=0.05, min=0, max=0.1, step=0.01, description='g'))


