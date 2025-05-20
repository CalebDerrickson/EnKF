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
    T = 1.0
    dts = [2, 4, 5, 8].*0.001
    Nts = [Int(T / dt) for dt in dts]
    ks = 1:10
    
    for i in 1:length(Nts)
        lines = zeros(1+2*length(ks), Nts[i])
        offset = 1
        lines[offset, :] .= DoAnalysis(Nts[i], localization, ks[1], dts[i], seed)
        writetofile(seed, dts[i], lines[1, :])
        

        offset += length(ks)
        for k in ks
            lines[offset+k, :] .= DoAnalysis(Nts[i], TwoVecchia, k, dts[i], seed)
            writetofile(seed, dts[i], lines[offset+k, :])
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
    
    c_ensemble_idx = shuffle(1:N*N)   
    c_ensemble = [[XYGrid[1][i], XYGrid[2][i]] for i in c_ensemble_idx]   
    ptGrid = c_ensemble


    # params = [σ, ρ, ν]
    params = [1.0, 0.8, 2.25]
    MatCov = VecchiaMLE.generate_MatCov(N, params, ptGrid)
    xf = Matrix{Float64}(VecchiaMLE.generate_Samples(MatCov, N, Ne)')
    xf .+= repeat(reshape(u, N*N, 1), 1, Ne)

    # Observations
    observe_index = sample(1:N*N, Int(0.25*N*N); replace=false)
    sigma = 0.1
    R = Diagonal(fill(sigma^2, length(observe_index)))
    H = view(Matrix{Float64}(I, N*N, N*N), observe_index, :)

    # Covariance localization
    infl = 1.01
    localization_radius = 0.3
    rho = cal_rho(localization_radius, N*N, gaspari_cohn, N, Lx, Ly)


    for i = 1:Nt
        # Propagate truth
        u = forward_euler(u, N, dx, dy, dt, cx, cy, nu)

        # Propagate ensembles
        for j = 1:N
            temp = forward_euler(xf[:, j], N, dx, dy, dt, cx, cy, nu)
            xf[:, j] = reshape(temp, N*N, 1)
        end

        # Observation vector from truth
        y = reshape(u, N*N, 1)[observe_index, :]
        
        # Do the EnKF analysis, updates xf
        BabyKF(xf, y, H, R, infl, rho, ptGrid, observe_index, strat, k)
        
        if strat != localization
            open("EnKF_output_diag_$(dt).txt", "a") do io
                X_cov = 1/(Ne-1) * sum(xf[:, j] * xf[:, j]' for j in 1:size(xf, 2))
                line = [X_cov[j, j] for j in 1:size(X_cov, 2)]
                writedlm(io, [line])
            end
        end
        temp_analysis_mean = mean(xf, dims=2)
        
    
        # Compute and save RMS error
        res[i] = (1 / N) * norm(temp_analysis_mean .- reshape(u, N*N, 1))
        
        logstatus(strat, k, i, res[i], seed, dt)
        if res[i] > 1e2 break end
    end

    return res
end


function logstatus(strat::Strategy, k::Int, i::Int, num::Float64, seed::Int, dt::Float64)
    output = ""
    if strat == localization output *= "0"
    elseif strat == OneVecchia output *= "1"
    elseif strat == TwoVecchia output *= "2" 
    end
    line = ["$(output), k = $(k), Step = $(i), rms = $(num)"]
    println(line[1])
    open("EnKF_log_$(seed)_$(dt).txt", "a") do io
        writedlm(io, line)
    end
end

function writetofile(seed, dt, line)
    open("EnKF_output_$(seed)_$(dt).txt", "a") do io
        writedlm(io, [line])
    end
end