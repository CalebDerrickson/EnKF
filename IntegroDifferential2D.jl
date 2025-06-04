using Plots
using LinearAlgebra
using VecchiaMLE

function main()
    dt = 0.005
    #α = 0.75
    T = 10.0
    Nt = Int(T / dt)

    # Grid and physical setup
    N = 50

    GridLen = min(0.2 * N, 2.0)
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
    centers = [(Lx / 2, Ly / 2)]
    widths = [1.0]
    amps = [1.0]

    init = zeros(N, N)
    for i in eachindex(centers)
        init .+= amps[i] .* exp.(-((X .- centers[i][1]).^2 .+ (Y .- centers[i][2]).^2) ./ (2 * widths[i]^2))
    end

    #init ./= sum(init) 

    state[:, 1] .= reshape(init, N*N, 1)


    # Define the nonlinearity f(N)
    r = 0.8
    K = 1.0
    #f(s) = r * clamp.(s, 0.0, K*2) .* (1 .- clamp.(s, 0.0, K*2) ./ K)
    f(s) = r .* s .* (1 .- s./ K)


    # kernel matrix K(x, y)
    params = [1.0, 0.25, 2.5]
    kernel = Matrix{Float64}(VecchiaMLE.generate_MatCov(N, params, ptGrid))
    for i in 1:size(kernel, 1)
        kernel[i, :] ./= sum(kernel[i, :])
    end
    #kernel .*= dx * dy

    # Time stepping
    for t in 1:Nt
        state[:, t+1] = state[:, t] .+ dt .* (kernel * f(state[:, t]))
        #state[:, t+1] ./= maximum(state[:,t+1])
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