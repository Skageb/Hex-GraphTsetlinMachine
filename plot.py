def plot_lines(y_values, x_values=None, x_label='', y_label='', title='', legends=None, path=None):
    """
    Plots line graphs from one or more arrays. (This function was generated with GPT-o1-preview)

    Parameters:
    ----------
    y_values : array-like or list of array-like
        The y-values to plot. Can be a single array or a list of arrays for multiple lines.
    x_values : array-like or list of array-like, optional
        The x-values corresponding to y-values. If None, defaults to indices starting from 1.
        If provided, should match the length of y_values.
    x_label : str, optional
        Label for the x-axis.
    y_label : str, optional
        Label for the y-axis.
    title : str, optional
        Title of the plot.
    legends : list of str, optional
        Legend labels for each line. If None, defaults to 'Line 1', 'Line 2', etc.
    path : str, optional
        File path to save the plot image. If provided, the plot is saved to this path.

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.

    Example:
    -------
    # Plot a single line
    y = [1, 2, 3, 4, 5]
    fig = plot_lines(y, y_label='Value', title='Single Line Plot')

    # Plot multiple lines with custom x-values and legends
    y1 = [1, 2, 3, 4, 5]
    y2 = [2, 3, 4, 5, 6]
    x = [0, 1, 2, 3, 4]
    fig = plot_lines([y1, y2], x_values=[x, x], legends=['Dataset 1', 'Dataset 2'], 
                     x_label='Time', y_label='Value', title='Multiple Lines')
    """
    import matplotlib.pyplot as plt

    # Ensure y_values is a list
    if not isinstance(y_values, list):
        y_values = [y_values]

    # Handle x_values
    if x_values is None:
        x_values = [range(1, len(y)+1) for y in y_values]
    else:
        if not isinstance(x_values, list):
            x_values = [x_values]
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have the same number of elements.")
        # Check that each x and y pair have the same length
        for x, y in zip(x_values, y_values):
            if len(x) != len(y):
                raise ValueError("Each x and y pair must have the same length.")

    # Handle legends
    if legends is None:
        legends = [f'Line {i+1}' for i in range(len(y_values))]
    else:
        if len(legends) != len(y_values):
            raise ValueError("Length of legends must match number of lines.")

    # Create the plot
    fig, ax = plt.subplots()
    for x, y, legend in zip(x_values, y_values, legends):
        ax.plot(x, y, label=legend)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if legends:
        ax.legend()
    ax.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the plot if path is provided
    if path:
        plt.savefig(path)

    return fig


import json


def plot_accuracy_across_epochs(json_file_path, output_path=None):
    """
    Loads a JSON file containing "train_accuracy_across_epochs" data and plots it using the plot_lines function.

    Parameters
    ----------
    json_file_path : str
        The path to the JSON file containing accuracy data.
    output_path : str, optional
        The path where the plot image should be saved. If None, the plot is not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    """
    # Load JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the training accuracies
    accuracies = data.get("train_accuracy_across_epochs", [])
    if not accuracies:
        raise ValueError("No 'train_accuracy_across_epochs' data found in the provided JSON file.")
    
    # Use the provided plot_lines function
    print(len(accuracies), len(range(1, len(accuracies) + 1)))
    fig = plot_lines(
        y_values=[accuracies],
        x_values=[range(1, len(accuracies) + 1)],
        x_label='Epoch',
        y_label='Accuracy (%)',
        title='Accuracy Across Epochs for 13x13 Games',
        legends=['Test Accuracy'],
        path=output_path
    )
    
    return fig

import matplotlib.pyplot as plt


def plot_accuracy_across_epochs_stop_marked(json_file_path, output_path=None):
    """
    Loads a JSON file containing "train_accuracy_across_epochs" data and plots it using matplotlib.
    Additionally, it marks the final data point and annotates it if near 100%, placing the annotation below the point.
    """
    # Load JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract the training accuracies
    accuracies = data.get("train_accuracy_across_epochs", [])
    if not accuracies:
        raise ValueError("No 'train_accuracy_across_epochs' data found in the provided JSON file.")

    # Prepare data for plotting
    x = list(range(1, len(accuracies) + 1))
    y = accuracies

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot line with a marker on the last point
    ax.plot(x, y, label='Test Accuracy', marker='o', markevery=[-1], markersize=6, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Across Epochs for 13x13 Hex Games')
    ax.grid(True)
    ax.legend()

    # Annotate the final point if it's near 100%
    final_x, final_y = x[-1], y[-1]
    if final_y > 99:
        annotation_text = f"{final_y:.2f}% Accuracy"
        # Place the text and arrow below the final point
        ax.annotate(
            annotation_text,
            xy=(final_x, final_y),
            xycoords='data',
            xytext=(final_x, final_y - 5),  # Shift text 5 units below the point
            textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
            horizontalalignment='center',
            verticalalignment='top'
        )

    plt.tight_layout()

    # Save the plot if path is provided
    if output_path:
        plt.savefig(output_path)

    return fig




if __name__ == '__main__':
    path = '/home/skage/projects/ikt457_learning-systems/tsetlin_machine/hex_project/skage_sandbox/results/13x13_board_0_turns_before_win.json'
    fig = plot_accuracy_across_epochs_stop_marked(path, '13x13_0turns_accuracy.png')