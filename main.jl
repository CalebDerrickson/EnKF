using Plots, PlotlyJS
using Random
using DelimitedFiles
using LinearAlgebra
using PProf
using VecchiaMLE
using SparseArrays

function main()
    seed = 8001
    Random.seed!(seed)
    dt = 0.001
    T = 0.01
    Nt = Int(T * 1.0/dt)
    
    ks = 1:10
    
    lines = zeros(1+length(ks), Nt)
    lines[1, :] = DoAnalysis(Nt, true, ks[1])
    
    for k in ks
        lines[k+1, :] = DoAnalysis(Nt, false, k)
    end


    outfile = "results_seed_$(seed).csv"
    writedlm(outfile, lines, ',')
    #lines = readdlm(outfile, ',')
    
    #plotting(lines, ks)
end

function DoAnalysis(Nt, localize::Bool, k)
    # Grid parameters
    N = 100
    Lx = 4.0
    Ly = 4.0
    dx = Lx / (N - 1)
    dy = Ly / (N - 1)

    
    # Time parameters
    dt = 0.001
    res = zeros(Nt)
    
    # Physical parameters
    cx = 1.0
    cy = 1.0
    nu = 0.01
    

    # create Grid
    x = LinRange(0.0, Lx, N)
    y = LinRange(0.0, Ly, N)

    X = [x for _ in y, x in x]
    Y = [y for y in y, _ in x]
    XYGrid = [X[:], Y[:]]

    centers = [0.25, 0.25]
    u = exp.(-100.0.*((X .- centers[1]).^2 .+ (Y .- centers[2]).^2))
    
    # Initialize the ensembles
    c_ensemble = [0.25; 0.25] .+ 0.1 * randn(2, N)

    xf = zeros(N*N, N)

    for i in 1:N
        xf[:, i] = reshape(exp.(-50*((X .- c_ensemble[1, i]).^2 + (Y .- c_ensemble[2, i]).^2)), N*N, 1) 
    end

    # Observation
    observe_index = 1:cld(N*N, 100):N * N
    H = view(Matrix{Float64}(I, N*N, N*N), observe_index, :)
    sigma = 0.01
    #R = (sigma^2) * I(size(H, 1));
    R = Diagonal(fill(sigma^2, size(H, 1)))

    infl = 1.01
    rms_value = 0.0

    #rep = zeros(N*N, length(observe_index))
    #L = zeros(N*N, N*N)
    ptGrid = VecchiaMLE.generate_safe_xyGrid(N)
    localization_radius = 0.3
    rho = cal_rho(localization_radius, N*N, gaspari_cohn, N, Lx, Ly)
    # iterate over time
    for i = 1:Nt
        # this is a twin experiment where we assume we have the "truth" and we
        # compare that with the analysis. "Truth" trajectory might not be
        # available in real life scenarios
    
        # propogate the "truth"
        u = forward_euler(u, N, dx, dy, dt, cx, cy, nu);
    
        # propogate each ensembles through the model for one time step
        for j = 1:N
            temp = forward_euler(view(xf, :, j), N, dx, dy, dt, cx, cy, nu);
            xf[:, j] .= reshape(temp, N*N, 1);
        end
    
        # create observations based around the truth
        y = view(reshape(u, N*N, 1), observe_index, :); # this can be non-linearized

        # Do the analysis
        temp_analysis = BabyKF(xf, y, H, R, infl, rho, ptGrid, observe_index, localize, k)#, rep, L);
        temp_analysis_mean = mean(temp_analysis, dims=2);
    
        res[i] = sqrt((((norm(temp_analysis_mean - reshape(u, N*N,1),2))^2) + (i == 1 ? 0.0 : res[i-1]^2)*(i-1)*length(reshape(u, N*N,1)))/(i*length(reshape(u, N*N,1))))
        println("Step = $i, rms = $(res[i])");    
        #fill!(rep, 0.0)
        #fill!(L, 0.0)
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

