import torch

import matplotlib.pyplot as plt

def intersections(basis: torch.Tensor, bias: torch.Tensor):
    Q_A, _ = torch.linalg.qr(basis)
    directions = Q_A @ -Q_A.T

    origin = torch.tensor([[-1., -1, -1]])
    
    cos_angles = origin@directions / (origin.norm(dim=-1)*directions.norm(dim=-1))
    angles = torch.arccos(cos_angles)
    to_ininity = abs(angles) > torch.pi/4


    print(angles*180/torch.pi)
    print(to_ininity)




def plot_plane(basis=None, bias=None, relu=True):
    if basis is None:
        basis  = torch.tensor([[-1, 0], 
                            [1, 1], 
                            [1, -1]], dtype=torch.float32)
    if bias is None:
        bias = torch.tensor([1,1,1])
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    # Create a grid for the plane (e.g., xy-plane)
    x = torch.linspace(-5, 5, 50)
    y = torch.linspace(-5, 5, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    values = torch.stack((X, Y), dim=-1) @ basis.T + bias
    X, Y, Z = values[..., 0], values[..., 1], values[..., 2]

    
    # Plot the surface
    ax.plot_surface(X, Y, Z, alpha=0.7)

    if relu:
        values = torch.relu(values)
        X, Y, Z = values[..., 0], values[..., 1], values[..., 2]
        ax.plot_surface(X, Y, Z, alpha=0.7)


    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    #ax.grid(True)
    
    # Set axis ticks
    ax.scatter([0], [0], [0], color='black', s=100, marker='o', label='Origin')
    ax.legend()

    # Rotate the view (elevation, azimuth)
    ax.view_init(elev=20, azim=45)

def plot_line(basis=None, bias=None):
    if basis is None:
        basis = torch.tensor([1,1.,2])
    if bias is None:
        bias = torch.tensor([0,2.,2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    t = torch.linspace(-5, 5, 100)
    line = bias + t.unsqueeze(-1) * basis
    
    ax.plot(line[:, 0], line[:, 1], line[:, 2], 'b-', linewidth=2)

    line = torch.relu(line)
    ax.plot(line[:, 0], line[:, 1], line[:, 2], 'r-', linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)
    
    ax.scatter([0], [0], [0], color='black', s=100, marker='o', label='Origin')
    ax.legend()

    Q, R = torch.linalg.qr(basis.unsqueeze(0).T, mode='complete')
    ker_A = Q[:, 1:]
    print(ker_A)
    print(basis@ker_A)
    x = torch.linspace(-5, 5, 100)
    y = torch.linspace(-5, 5, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij' )
    values = torch.stack((X, Y), dim=-1) @ ker_A.T + bias
    X, Y, Z = values[..., 0], values[..., 1], values[..., 2]

    # Plot the surface
    ax.plot_surface(X, Y, Z, alpha=0.5)

    for i in range(ker_A.shape[1]):
        v = ker_A[:, i]
        ax.quiver(
            bias[0], bias[1], bias[2],   # origin of the vector
            v[0], v[1], v[2],            # direction
            color=['green', 'orange'][i],
            linewidth=2,
            arrow_length_ratio=0.1,
            length=2,                    # scale for visibility
            label=f'ker basis {i+1}'
        )

def main():
    torch.manual_seed(8)
    bases = torch.rand(3, 2)
    bias = torch.rand(3)
    bases  = torch.tensor([[-1, 0], 
                            [-2, 1], 
                            [1, -1]], dtype=torch.float32)

    bias = torch.tensor([1,1,1])
    intersections(bases, bias)

    plot_plane(bases, bias)
    plt.show()
    plot_line()
    plt.show()

if __name__ == "__main__":
    main()
    
