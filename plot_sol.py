def plot_sol(sol):
    """
    Plot the solution of the ODE system.
    
    Parameters:
    sol : OdeResult
        The result of the ODE integration containing time and state data.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=f'x{i+1}(t)')
    plt.xlabel('Time t')
    plt.ylabel('States')
    plt.legend()
    plt.grid(True)
    plt.title('State Response')
    plt.show()
