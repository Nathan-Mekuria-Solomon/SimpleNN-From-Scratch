import matplotlib
import matplotlib.pyplot as plt
import argparse                     # to handle command-line input
import ast                          # explanation???

# matplotlib.use('QtAgg')

def draw_neural_networks(layer_sizes): # e.g. [2, 3, 4] : 3 layers
    """
    Draws a representative neural network.
    Args:
        layer_sizes (List) : List of integers where each integer represents number of neuron in that layer.
    """
    if not layer_sizes or len(layer_sizes) < 2:
        print("Error: at least two layers are needed (input, output).")
        return
    
    # Configuration for drawing
    v_spacing = 1.0
    total_layers = len(layer_sizes)
    total_width = 16.0

    # Dynamic horizontal spacing calculation
    # Number of gaps = (total_layers - 1)
    if total_layers > 1:
        h_spacing = total_width / (total_layers - 1)
    else:
        h_spacing = 0.0

    neuron_coords = []

    # Calculate coordinates
    for i, num_neurons in enumerate(layer_sizes):
        # Horizontal
        x = i * h_spacing

        # Vertical
        # Find center
        layer_center = (num_neurons - 1) * v_spacing / 2
        # Calculate y coordinates: Position (y * v_spacing) - center offset
        y_coords = [(y * v_spacing) - layer_center for y in range(num_neurons)]

        # Store coordinates
        neuron_coords.append(list(zip([x] * num_neurons, y_coords)))

    # Adjust figure size
    max_neurons = max(layer_sizes)
    fig, ax = plt.subplots(figsize= (total_width * 0.5, max_neurons * v_spacing * 0.8 + 2))

    # Draw connections
    for i in range(len(layer_sizes) - 1):
        source_layer = neuron_coords[i]
        target_layer = neuron_coords[i + 1]

        for sx, sy in source_layer:
            for tx, ty in target_layer:
                ax.plot([sx, tx], [sy, ty], 'r-', alpha= 0.4, linewidth= 0.5)

    # Draw neurons
    for i, layer in enumerate(neuron_coords):
        x_coords, y_coords = zip(*layer)

        plt.plot(x_coords, y_coords,
                 s= 1000,
                 color= 'lightblue',
                 edgecolor= 'blue',
                 zorder= 3                  # Explanation ??
                 )
        
        # Add layer labels
        layer_type = ["Input", "Hidden", "Output"][min(i, 2) if total_layers > 2 else i]
        ax.text(x_coords[0], max(y_coords) + 0.5, 
                f'{layer_type} Layer\n({len(layer)})',              # Explanation ???
                ha='center', fontsize=12, fontweight='bold')
        
    # Final Plot Customization
    ax.set_title('Dense Neural Network Structure with Even Spacing', fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Ask user for input
    layer_structure = input("Enter the layer structure as a list, e.g. [3, 4, 2]: ")

    try:
        # Safely evaluate the input (turn string "[3,4,2]" into Python list)
        layer_sizes = ast.literal_eval(layer_structure)

        # Validate
        if not isinstance(layer_sizes, list) or not all(isinstance(n, int) and n > 0 for n in layer_sizes):
            raise ValueError("All values of layers must be positive integers")
    
        # Call your visualization function
        draw_neural_networks(layer_sizes)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")