import matplotlib.pyplot as plt

def plot_coefficient_evolution(beta_history_arr, selected_indices, total_features):
    """
    Creates a 1-column, 3-row subplot:
      - Combined view: All features (selected in green, not selected in gray).
      - Unselected features only.
      - Selected features only.
    
    All subplots share the same y-axis.
    """
    n_selected = len(selected_indices)
    n_not_selected = total_features - n_selected

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18), sharey=True)

    # Subplot 1: Combined view.
    ax = axes[0]
    for i in range(total_features):
        if i in selected_indices:
            ax.plot(beta_history_arr[:, i], color='green', linewidth=0.5, alpha=0.5)
        else:
            ax.plot(beta_history_arr[:, i], color='gray', linewidth=0.5, alpha=0.5)
    ax.plot([], [], color='green', linewidth=1, label=f'Selected features ({n_selected})')
    ax.plot([], [], color='gray', alpha=0.3, label=f'Not elected ({n_not_selected})')
    ax.set_title("Combined: All Features")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Coefficient Value")
    ax.legend(loc="best", fontsize='small', ncol=2)

    # Subplot 2: Unselected features.
    ax = axes[1]
    for i in range(total_features):
        if i not in selected_indices:
            ax.plot(beta_history_arr[:, i], color='gray', linewidth=0.5, alpha=0.5)
    ax.set_title(f"Unselected Features ({n_not_selected})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Coefficient Value")

    # Subplot 3: Selected features.
    ax = axes[2]
    for i in selected_indices:
        ax.plot(beta_history_arr[:, i], color='green', linewidth=0.5, alpha=0.5)
    ax.set_title(f"Selected Features ({n_selected})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Coefficient Value")

    plt.tight_layout()
    plt.show()
