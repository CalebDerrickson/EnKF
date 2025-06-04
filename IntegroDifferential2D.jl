using Plots
using LinearAlgebra
using VecchiaMLE

function main()
    dt = 0.001
    #α = 0.75
    T = 0.5
    Nt = Int(T / dt)

    # Grid and physical setup
    N = 50

    GridLen = max(0.2 * N, 10.0)
    Lx = Ly = GridLen 

    dx, dy = Lx / (N - 1), Ly / (N - 1)

    # Generate grid
    ptGrid = VecchiaMLE.generate_safe_xyGrid(N)
    x = LinRange(0.0, Lx, N)
    y = LinRange(0.0, Ly, N)
    X = [x for _ in y, x in x]
    Y = [y for y in y, _ in x]

    # Initialize state
    # States are column vectors!
    state = zeros(N*N, Nt+1)


    # Initial truth. Gaussian centered at center
    centers = [Lx / 2, Ly / 2]

    state[:, 1] .= reshape(exp.(-1.0 .* ((X .- centers[1]).^2 .+ (Y .- centers[2]).^2)), N*N, 1)


    # Define the nonlinearity f(N)
    r = 2.0
    K = 20.0
    f(s) = r * clamp.(s, 0.0, K*2) .* (1 .- clamp.(s, 0.0, K*2) ./ K)

    # Precompute kernel matrix K(x, y)
    #kernel = zeros(length(ptGrid), length(ptGrid))
    #for i in eachindex(ptGrid)
    #    for j in eachindex(ptGrid)
    #        kernel[i, j] = exp(-α * norm(ptGrid[i] - ptGrid[j])^2)
    #    end
    #    kernel[i, :] ./= sum(kernel[i, :])
    #end
    #display(kernel[1, :])
    #kernel .*= (0.5 * α * dx * dy)   # includes prefactor and integration weight
    params = [1.0, 0.8, 2.25]
    kernel = Matrix{Float64}(VecchiaMLE.generate_MatCov(N, params, ptGrid))
    kernel ./= maximum(kernel)
    kernel .*= dx * dy

    # Time stepping
    for t in 1:Nt
        state[:, t+1] = state[:, t] .+ dt .* (kernel * f(state[:, t]))
    end


    # Visualization
    plot_idxs = round.(Int, range(start=1, stop=Nt, length=4))[2:end]
    l = Plots.@layout [a b ; c d]
    plots = Vector{Plots.Plot{Plots.GRBackend}}(undef, length(plot_idxs)+1)
    plots[1] = heatmap(x, y, reshape(state[:, 1], N, N), xlabel="X", ylabel="Y", title="Timestep 0")
    
    display(state)
    for (k, i) in enumerate(plot_idxs)
        plots[k+1] = heatmap(x, y, reshape(state[:, i], N, N), xlabel="X", ylabel="Y", title="Timestep $i")
    end
    plot(plots..., layout=l)
end