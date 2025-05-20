function BabyKF(xf, y, H, R, infl, rho, ptGrid::AbstractVector, observe_index::AbstractVector, strat::Strategy, k)
    num_states, N = size(xf) # 2500 x 50 
    num_observation = size(y, 1)

    # Add noise to observation
    y_perturbed = repeat(y, 1, N) + sqrt.(R) * randn(num_observation, N)

    # Add inflation
    xfm = mean(xf, dims=2)
    xf_dev = (1 / sqrt(N-1)) * (xf .- repeat(xfm, 1, N))
    xf .= sqrt(N - 1) * sqrt(infl) * xf_dev + repeat(xfm, 1, N)
    
    # Using inn as the observed update of kalman filter
    inn = y_perturbed .- view(xf, observe_index, :)

    if strat == localization 
        Localization(xf, y, xf_dev, inn, observe_index, rho, R)
    elseif strat == OneVecchia
        OneVecchiaEnKF(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k)
    elseif strat == TwoVecchia
        TwoVecchiaEnKF(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k)
    end
end

function Localization(xf, y, xf_dev, inn, observe_index, rho, R)
    num_states, N = size(xf) # 2500 x 50 
    num_observation = size(y, 1)

    rhoPH = rho[:, observe_index]
    rhoHPHt = rho[observe_index, observe_index]

    zb = xf[observe_index, :]
    zbm = mean(zb, dims=2) 
    Zb = (1 / sqrt(N - 1)) * (zb .- repeat(zbm, 1, N))

    K = (rhoHPHt .* (Zb * Zb') .+ R) \ inn
    xf .+= rhoPH .* (xf_dev * Zb') * K
    return nothing
end

function OneVecchiaEnKF(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k)
    num_states, N = size(xf)
    num_observation = size(y, 1)

    # Have to transpose them since that's how VecciaMLE parses them. 
    xf_mat = Matrix{Float64}(xf')
    xfm = mean(xf_mat, dims = 2) # mean by row. 
    xf_mat .-= repeat(xfm, 1, num_states)
    
    n = Int(sqrt(size(xf_mat, 2)))
    input = VecchiaMLEInput(n, k, xf_mat, N, 5, 1; ptGrid=ptGrid)
    
    L = VecchiaMLE_Run(input)[2]
    rep = L' \ (L \ H')
    xf .+= rep * ((view(rep, observe_index, :).+R) \ inn)
    return L
end


function TwoVecchiaEnKF(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k)
    num_states, Ne = size(xf)
    num_observation = size(y, 1)

    # Have to transpose them since that's how VecciaMLE parses them. 
    xf_mat = Matrix{Float64}(xf')
    xfm = mean(xf_mat, dims = 2) # mean by row. 
    xf_mat .-= repeat(xfm, 1, num_states)
    
    n = Int(sqrt(size(xf_mat, 2)))
    input = VecchiaMLEInput(n, k, xf_mat, Ne, 5, 1; ptGrid=ptGrid, skip_check=true)

    L = LinearAlgebra.LowerTriangular(VecchiaMLE_Run(input)[2])
    
    # Generate Randomness from normal
    R_half_Z = randn(num_observation, Ne)
    R_half_Z = sqrt.(R) * R_half_Z # Since R is diagonal this is fine
    
    chol = xf_mat[:, observe_index]
    
    chol .+= R_half_Z'
    subptGrid = ptGrid[observe_index]
    n = Int(sqrt(size(chol, 2)))

    samples = chol * randn(num_observation, num_observation)

    input = VecchiaMLEInput(n, k, samples, Ne, 5, 1; ptGrid=subptGrid, skip_check=true)
    S = LinearAlgebra.LowerTriangular(VecchiaMLE_Run(input)[2])
    
    # Next form kalman filter
    K = L' \ (L \ H')
    K .= K * S * S'
    
    # Next perform update
    xf .+= K * inn
    return 
end