using Plots
using Random
using DelimitedFiles
using LinearAlgebra
using PProf
using VecchiaMLE


function main()
    seed = 7763
    Random.seed!(seed)
    T = 1.0
    dts = [8].*0.001
    Nts = [Int(T / dt) for dt in dts]
    ks = 1:10
    OdeMethod::ODEMethod = Integro

    for i in 1:length(Nts)
        
        # line = DoAnalysis(Nts[i], localization, ks[1], dts[i], seed, OdeMethod)
        # writetofile(seed, dts[i], view(line, 1:Nts[i]))
        # GC.gc() # okay to run gc here? 

        
        for k in ks
          line = DoAnalysis(Nts[i], OneVecchia, k, dts[i], seed, OdeMethod)
          writetofile(seed, dts[i], view(line, 1:Nts[i]))
           GC.gc() # okay to run gc here? 
        end

        # for k in ks
            # line = DoAnalysis(Nts[i], TwoVecchia, k, dts[i], seed, OdeMethod)
            # writetofile(seed, dts[i], view(line, 1:Nts[i]))
            # GC.gc() # okay to run gc here? 
        # end

    end
    
end

function DoAnalysis(Nt, strat::Strategy, k, dt, seed, OdeMethod::ODEMethod=ForwardEuler)
    # Grid and physical setup
    N = 50
    num_states = N*N

    GridLen = max(0.2 * N, 10.0)
    Lx = Ly = GridLen 
    
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
    X = Vector{Float64}(reshape(X', num_states))
    Y = Vector{Float64}(reshape(Y', num_states))

    # Initial truth
    centers = [Lx / 2, Ly / 2]
    u = exp.(-100.0 .* ((X .- centers[1]).^2 .+ (Y .- centers[2]).^2))

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
    #observe_index = sort(sample(1:N*N, Int(0.25*N*N); replace=false))
    observe_index = vec( reshape(1:N*N, N, N)[1:2:end, 1:2:end] )

    # sigma is very important for 1Vecchia. Only really tested for ForwardEuler.
    # Blows up for small values, like 0.25. Decent value is 3.125.
    # numerical instability? Size of the grid? Sharpness of the initial truth?
    sigma = 3.125
    R = Diagonal(fill(sigma^2, length(observe_index)))
    H = view(Matrix{Float64}(I, N*N, N*N), observe_index, :)

    # kernel matrix K(x, y)
    if OdeMethod == Integro
        params = [1.0, 0.8, 2.5]
        kernel = Matrix{Float64}(VecchiaMLE.generate_MatCov(N, params, ptGrid))
        for i in 1:size(kernel, 1)
            kernel[i, :] ./= sum(kernel[i, :])
        end
    end
    

    # Covariance localization
    infl = 1.01
    if strat == localization
        localization_radius = 0.3
        rho = cal_rho(localization_radius, N*N, gaspari_cohn, N, Lx, Ly, X, Y)
    end
    PATTERN_CACHE = PatternCache(nothing, nothing)


    for i = 1:Nt
        # Propagate truth
        if OdeMethod == ForwardEuler
            u = forward_euler(u, N, dx, dy, dt, cx, cy, nu)
        else
            u = integro(u, N, dt, kernel)
        end

        # Propagate ensembles
        for j = 1:Ne
            if OdeMethod == ForwardEuler
                temp = forward_euler(xf[:, j], N, dx, dy, dt, cx, cy, nu)
                xf[:, j] .= reshape(temp, N*N, 1)
            else
                temp = integro(xf[:, j], N, dt, kernel)
                xf[:, j] .= reshape(temp, N*N, 1)
            end
        end

        # Observation vector from truth
        y = reshape(u, N*N, 1)[observe_index, :]
        
        # Do the EnKF analysis, updates xf
        if strat == localization
            BabyKF(xf, y, H, R, infl, rho, ptGrid, observe_index, strat, k, PATTERN_CACHE)
        else
            BabyKF(xf, y, H, R, infl, 0.0, ptGrid, observe_index, strat, k, PATTERN_CACHE)
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