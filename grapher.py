import matplotlib.pyplot as plt

linestyles = [
    '-',                        # solid
    '--',                       # dashed
    ':',                        # dotted
    '-.',                       # dash-dot
    (0, (8, 3)),                # long dash
    (0, (1, 4)),                # loosely dotted
    (0, (3, 1, 1, 1, 1, 1)),    # dash-dot-dot
    (0, (5, 8)),                # loosely dashed
    (0, (12, 3, 2, 3)),         # long dash-dot
    (0, (2, 1)),                # densely dashed
]

def create_graphs(met1, met2, met1_name="", met2_name="", plot_title=None, plot_file="", log_freq=64, show=False):
    if isinstance(met1[0], list):
        x = range(0, len(met1[0])*log_freq, log_freq)  
    else:
        x = range(0, len(met1)*log_freq, log_freq)  
    fig, ax1 = plt.subplots(figsize=(12,12))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(met1_name, color=color)
    if isinstance(met1[0], list):
        for i, met in enumerate(met1):
            color = (0.5 + 0.5*((i+1)/len(met1)), 0, 0)
            ax1.plot(x, met, color=color, linestyle=linestyles[i%len(linestyles)])
    else:
        ax1.plot(x, met1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    if plot_title is not None:
        ax1.set_title(plot_title)

    ax2 = ax1.twinx()  

    color = 'tab:blue'
    ax2.set_ylabel(met2_name, color=color)  
    if isinstance(met2[0], list):
        largest = float("-inf")
        for i, met in enumerate(met2):
            color = (0, 0.5 + 0.5*((i+1)/len(met1)), 0)
            ax2.plot(x, met, color=color, linestyle=linestyles[i%len(linestyles)])
            largest = max(largest, max(met))
        # ax1
            
    else:
        ax1.plot(x, met2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() 
    
    plt.savefig(f"{plot_file}_{met1_name}_{met2_name}.png")
    if show:
        plt.show()
    


def main():
    lines = s.splitlines()
    loss_lines = [l for l in lines if "Loss" in l]
    collapse_lines = [l for l in lines if "Collapsed" in l]
    accuracy_lines = [l for l in lines if "Accuracy" in l]
    get_val = lambda line: float(line.split(":")[-1].replace("%",""))
    losses = list(map(get_val, loss_lines))
    collapses = list(map(get_val, collapse_lines))
    accuracys = list(map(get_val, accuracy_lines))
    
    x = range(len(losses))
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(x, accuracys, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Pairs Collapsing', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, collapses, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    
    
    plt.plot(range(len(losses)), collapses)
    plt.show()

if __name__ == "__main__":
    main()