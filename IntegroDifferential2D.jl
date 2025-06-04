#using Plots
using LinearAlgebra
using VecchiaMLE
using CairoMakie

function main()
    dt = 0.005
    #α = 0.75
    T = 10.0
    Nt = Int(T / dt)

    # Grid and physical setup
    N = 25

    GridLen = min(0.2 * N, 10.0)
    Lx = Ly = GridLen 

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
    centers = [(Lx / 4, Ly / 2), (Lx / 1.4, Ly / 1.4)]
    widths = [1.0, 0.4]
    amps = [0.6, 0.4]

    init = zeros(N, N)
    for i in eachindex(centers)
        init .+= amps[i] .* exp.(-((X .- centers[i][1]).^2 .+ (Y .- centers[i][2]).^2) ./ (2 * widths[i]^2))
    end

    #init ./= sum(init) 

    state[:, 1] .= reshape(init, N*N, 1)

    # Define the nonlinearity f(N)
    r = 0.8
    K = 1.0
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
        state[:, t+1] .= state[:, t] .+ dt .* (kernel * f(state[:, t]))
    end

    # The rest of this is just plotting 
    
    # Pick timesteps to visualize
    plot_idxs = round.(Int, range(start=1, stop=Nt, length=4))[2:end]
    all_maps = [reshape(state[:, i], N, N) for i in vcat(1, plot_idxs)]

    # Global color limits across all maps
    extremas = map(extrema, all_maps)
    global_min = minimum(first, extremas)
    global_max = maximum(last, extremas)
    clims = (global_min, global_max)

    # Layout dimensions
    n_rows = 2
    n_cols = 2

    fig = Figure(;size=(1920, 1080))

    for (k, map) in enumerate(all_maps)
        row = div(k - 1, n_cols) + 1
        col = mod(k - 1, n_cols) + 1
        ax = Axis(fig[row, col], title="Timestep $(k == 1 ? 0 : plot_idxs[k-1])", xlabel="X", ylabel="Y")
        heatmap!(ax, x, y, map; colorrange=clims)
    end

    # Shared colorbar on the right
    Colorbar(fig[:, n_cols + 1], limits=clims)

    fig
end