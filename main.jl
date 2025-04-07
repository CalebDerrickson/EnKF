
function main()

    # Grid parameters
    N = 50
    N = 50
    Lx = 1.0
    Ly = 1.0
    dx = Lx / (N - 1)
    dy = Ly / (N - 1)

    
    # Time parameters
    dt = 0.001
    T = 2.0
    Nt = round(T * 1.0/dt)
    
    
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
    u = exp.(-100*((X .- centers[1]).^2 .+ (Y .- centers[2]).^2))
    
    # Initialize the ensembles
    N = 50
    c_ensemble = [0.25; 0.25] .+ 0.1 * randn(2, N)

    xf = zeros(N*N, N)

    for i in 1:N
        xf[:, i] = reshape(exp.(-50*((X .- c_ensemble[1, i]).^2 + (Y .- c_ensemble[2, i]).^2)), N*N, 1) 
    end

    # Observation
    observe_index = 1:25:N * N
    H = Matrix{Float64}(I, N*N, N*N)
    H = H[observe_index, :]
    sigma = 0.01
    R = (sigma^2) * I(size(H, 1));
    

    infl = 1.01
    rms_value = 0

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
            temp = forward_euler(xf[:, j], N, dx, dy, dt, cx, cy, nu);
            xf[:, j] = reshape(temp, N*N, 1);
        end
    
        # create observations based around the truth
        y = H * reshape(u, N*N, 1); # this can be non-linearized
    
        # Do the analysis
        temp_analysis = BabyKF(xf, y, H, R, infl, rho, observe_index, false);
        temp_analysis_mean = mean(temp_analysis, dims=2);
    
        rms_value = sqrt(((norm(temp_analysis_mean - reshape(u, N*N,1), 2)).^2 +
            (rms_value^2)*(i-1)*length(reshape(u, N*N,1)))/(i*length(reshape(u, N*N,1))));
        println("Step = $i, rms = $(rms_value)");
    end
end