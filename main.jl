using Plots
using Random
using DelimitedFiles
using LinearAlgebra
using PProf
using VecchiaMLE
using SparseArrays


function main()
    seed = 6513
    Random.seed!(seed)
    dt = 0.001
    T = 0.1
    Nt = Int(T * 1.0/dt)
    
    ks = 1:10
    
    lines = zeros(1+length(ks), Nt)
    lines[1, :] .= DoAnalysis(Nt, true, ks[1], dt)
    
    for k in ks
        lines[k+1, :] .= DoAnalysis(Nt, false, k, dt)
    end


    outfile = "results_seed_$(seed).csv"
    writedlm(outfile, lines, ',')
    #lines = readdlm(outfile, ',')
    
    #plotting(lines, ks)
end

function DoAnalysis(Nt, localize::Bool, k, dt)
    # Grid and physical setup
    N = 50
    Lx, Ly = 10.0, 10.0
    dx, dy = Lx / (N - 1), Ly / (N - 1)
    cx, cy, nu = 1.0, 1.0, 0.01
    res = zeros(Nt)

    # Generate grid
    x = LinRange(0.0, Lx, N)
    y = LinRange(0.0, Ly, N)
    X = [x for _ in y, x in x]
    Y = [y for y in y, _ in x]
    XYGrid = [X[:], Y[:]]

    # Initial truth
    centers = [Lx / 2, Ly / 2]
    u = exp.(-100 .* ((X .- centers[1]).^2 .+ (Y .- centers[2]).^2))

    # Initial ensemble
    Ne = N  # ensemble size
    c_ensemble = rand(2, N*N) .* [Lx; Ly]
    xf = zeros(N*N, Ne)

    for i in 1:Ne
        xc, yc = c_ensemble[1, i], c_ensemble[2, i]
        gaussian = exp.(-50 .* ((X .- xc).^2 .+ (Y .- yc).^2))
        xf[:, i] = reshape(gaussian, N*N)
    end

    # Observations
    observe_index = 1:cld(N*N, 100):N*N
    sigma = 0.01
    R = Diagonal(fill(sigma^2, length(observe_index)))
    H = view(Matrix{Float64}(I, N*N, N*N), observe_index, :)

    # Covariance localization
    infl = 1.01
    localization_radius = 0.3
    rho = cal_rho(localization_radius, N*N, gaspari_cohn, N, Lx, Ly)
    rep = zeros(N*N, length(observe_index))
    L = zeros(N*N, N*N)
    ptGrid = [col for col in eachcol(c_ensemble)]

    # Setup truth propagation
    p_truth = ADParams(N, dx, dy, cx, cy, nu, zeros(N,N))
    prob_truth = ODEProblem(advection_diffusion!, u, (0.0, Nt*dt), p_truth)
    int_truth = init(prob_truth, Tsit5(), dt=dt, adaptive=false, save_everystep=false)

    # Setup ensemble propagation
    p_ens = ADParams(N, dx, dy, cx, cy, nu, zeros(N,N))
    prob_ens = ODEProblem(advection_diffusion!, zeros(N*N), (0.0, Nt*dt), p_ens)
    int_ens = init(prob_ens, Tsit5(), dt=dt, adaptive=false, save_everystep=false)

    for i = 1:Nt
        # Propagate truth
        reinit!(int_truth, u)
        step!(int_truth)
        u .= int_truth.u

        # Propagate ensembles
        for j in 1:Ne
            reinit!(int_ens, xf[:, j])
            step!(int_ens)
            xf[:, j] .= int_ens.u
        end

        # Observation vector from truth
        y = view(reshape(u, N*N, 1), observe_index, :)

        # Do the EnKF analysis
        temp_analysis = BabyKF(xf, y, H, R, infl, rho, ptGrid, observe_index, localize, k, rep, L)
        temp_analysis_mean = mean(temp_analysis, dims=2)
        #xf .= temp_analysis  # update ensemble

        # Compute and save RMS error
        res[i] = (1 / N) * norm(temp_analysis_mean .- reshape(u, N*N, 1))

        open("EnKF_output.txt", "a") do io
            println(io, "$(localize ? 0 : k),$i,$(res[i])")
        end
        println("Step = $i, rms = $(res[i])")

        fill!(rep, 0.0)
        fill!(L, 0.0)
    end

    return res
end

function plotting(res::AbstractMatrix, ks)
    len = size(res, 2)
    it = 1
    labels = ["Localize"]
    for k in 1:4
        push!(labels, "k = $(k)" )
    end
    p = plot()
    for row in 1:4
        plot!(p, 1:len, res[row, :], label=labels[it])
        it+=1
    end

    xlabel!("Iteration")
    ylabel!("RMSE")
    title!("Vecchia Neighbors")
    plot!(yscale=:log10, legend=:outerbottom, legendcolumns=3)
    display(p)

end

