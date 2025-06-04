using Plots

# Parameters
L = 10.0               # domain length
nx = 200               # number of spatial points
x = range(-L/2, L/2; length=nx)
dx = step(x)
α = 1              # kernel decay rate
nt = 25                # number of time steps

# Define the nonlinearity f(N)
r = 2.0
K = 20.0
f(N) = r * N * (1 - N / K)   # logistic-like nonlinearity

# Initialize N(x, t=0)
N = zeros(nx, nt+1)
N[:, 1] .=  exp.(-1.0.*x.^2) # small random initial population

# Precompute kernel matrix K(x, y)
kernel = zeros(nx, nx)
for i in 1:nx
    for j in 1:nx
        kernel[i, j] = exp(-α * abs(x[i] - x[j]))
    end
end
kernel .*= (0.5 * α * dx)   # includes prefactor and integration weight

# Time stepping
for t in 1:nt
    for i in 1:nx
        N[i, t+1] = sum(kernel[i, :] .* f.(N[:, t]))
    end
end

# Visualization
heatmap(x, 0:nt, N', xlabel="x", ylabel="Time", title="Evolution of N(x, t)", colorbar_title="N")