using Plots
using Random
using DelimitedFiles
using LinearAlgebra
using PProf
using VecchiaMLE
using SparseArrays


function main()
    seed = 4681
    Random.seed!(seed)
    dts = (1:5).*0.001
    Nt = 1000
    ks = 1:10
    
    lines = zeros(1+2*length(ks), Nt)
    for dt in dts
        lines[1, :] .= DoAnalysis(Nt, localization, ks[1], dt, seed)
        for k in ks
            lines[1+k, :] .= DoAnalysis(Nt, OneVecchia, k, dt, seed)
        end
        for k in ks
            lines[1+length(ks)+k, :] .= DoAnalysis(Nt, TwoVecchia, k, dt, seed)
        end
        open("EnKF_output_$(seed)_$(dt).txt", "a") do io
            writedlm(io, lines)
        end
    end

end

function DoAnalysis(Nt, strat::Strategy, k, dt, seed)
    # Grid and physical setup
    N = 50
    Lx, Ly = 10.0, 10.0
    dx, dy = Lx / (N - 1), Ly / (N - 1)

    # Transport speed and diffusion
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
    Ne = 100  # ensemble size
    # these centers needs to be on the grid, should be random indices, and then centers would be XYGrid[RandomIndices]. 
    #c_ensemble = [Lx/2; Ly/2] .+ 0.1 .* randn(2, N*N)
    c_ensemble_idx = shuffle(1:N*N)
    c_ensemble = [[XYGrid[1][i], XYGrid[2][i]] for i in c_ensemble_idx]   
    xf = zeros(N*N, Ne)
    ptGrid = c_ensemble

    # Generate a covariance matrix for the xf noise distrubition.
    #MatCov = VecchiaMLE.generate_MatCov(N, [5.0, 0.2, 2.25, 0.25], ptGrid)
    #xf = VecchiaMLE.generate_Samples(MatCov, N, Ne; mode = VecchiaMLE.cpu)'

    for i in 1:Ne
        xf[:, i] .= reshape(u.+ 0.1.*randn(size(u)), N*N, 1)
    end

    # Observations
    observe_index = sample(1:N*N, 25; replace=false) 
    sigma = 0.01
    R = Diagonal(fill(sigma^2, length(observe_index)))
    H = view(Matrix{Float64}(I, N*N, N*N), observe_index, :)

    # Covariance localization
    infl = 1.01
    localization_radius = 0.3
    rho = cal_rho(localization_radius, N*N, gaspari_cohn, N, Lx, Ly)

    

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
        y = reshape(u, N*N, 1)[observe_index, :]
        
        # Do the EnKF analysis, updates xf
        kl_div = BabyKF(xf, y, H, R, infl, rho, ptGrid, observe_index, strat, k)
        temp_analysis_mean = mean(xf, dims=2)

        # Compute and save RMS error
        res[i] = (1 / N) * norm(temp_analysis_mean .- reshape(u, N*N, 1))
        output = ""
        if strat == localization output *= "0,"
        elseif strat == OneVecchia output *= "1,"
        elseif strat == TwoVecchia output *= "2," 
        end

        println("$(output), k = $(k), Step = $(i), rms = $(res[i]), ")
        if res[i] > 1e4 break end
    end

    return res
end
