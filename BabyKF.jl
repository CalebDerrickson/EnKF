function BabyKF(xf, y, H, R, infl, rho, ptGrid::AbstractVector, observe_index::AbstractVector, strat::Strategy, k, PATTERN_CACHE)
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
        TwoVecchiaEnKF(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k, PATTERN_CACHE)
    elseif strat == Empirical
        EmpiricalAnalysis(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k)
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

function EmpiricalAnalysis(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k)

    zb = xf[observe_index, :]
    zbm = mean(zb, dims=2) 
    Zb = (1 / sqrt(N - 1)) * (zb .- repeat(zbm, 1, N))

    K = ( (Zb * Zb') .+ R) \ inn
    xf .+= (xf_dev * Zb') * K
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
    
    # The below code is straight Kalman Filter. 
    #rep = L' \ (L \ H')
    #xf .+= rep * ((view(rep, observe_index, :).+R) \ inn)

    # The below code is Phil's idea (SMW)
    # K = BₖHₖᵀ[Rₖ⁻¹ - Rₖ⁻¹Z(ZᵀRₖ⁻¹Z + I)⁻¹ZᵀRₖ⁻¹]
    # Z = 1 / √(N-1) * HₖX
    
    Z = xf[observe_index, :] ./ sqrt(N-1)
    K = L' \ (L \ H')
    K .= K * (inv(R) - (R\Z) * ((Z' / R * Z + I) \ Z') / R)
    # Next perform update
    xf .+= K * inn

    return 
end


function TwoVecchiaEnKF(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k, PATTERN_CACHE)
    _, Ne = size(xf)
    num_observation = size(y, 1)

    # Have to transpose them since that's how VecciaMLE parses them. 
    xf_mat = Matrix{Float64}(xf')
    xfm = mean(xf_mat, dims = 1) # mean by col. 
    xf_mat .-= repeat(xfm, Ne, 1)
    
    n = Int(sqrt(size(xf_mat, 2)))
    
    if PATTERN_CACHE.L === nothing
        input = VecchiaMLEInput(n, k, xf_mat, Ne, 5, 1; ptGrid=ptGrid, skip_check=true)
    else
        input = VecchiaMLEInput(n, k, xf_mat, Ne, 5, 1; ptGrid=ptGrid, skip_check=true,
            rowsL = PATTERN_CACHE.L.rowval,
            colsL = PATTERN_CACHE.L.colval,
            colptrL = PATTERN_CACHE.L.colptr
        )
    end

    L = VecchiaMLE_Run(input)[2] ./ sqrt(Ne)

    if PATTERN_CACHE.L === nothing
        cache_pattern!(L, :L, PATTERN_CACHE)
    end

    #L = LinearAlgebra.LowerTriangular(L)

    # Generate Randomness from normal
    R_half_Z = randn(num_observation, Ne)
    R_half_Z .= sqrt.(R) * R_half_Z # Since R is diagonal this is fine
    
    chol = xf_mat[:, observe_index]
    
    chol .+= R_half_Z'
    subptGrid = ptGrid[observe_index]
    n = Int(sqrt(size(chol, 2)))

    samples = randn(Ne, Ne) * chol 

    if PATTERN_CACHE.S === nothing
        input = VecchiaMLEInput(n, cld(k, 4), samples, Ne, 5, 1; ptGrid=subptGrid, skip_check=true)
    else
        input = VecchiaMLEInput(n, cld(k, 4), samples, Ne, 5, 1; ptGrid=subptGrid, skip_check=true, 
            rowsL = PATTERN_CACHE.S.rowval,
            colsL = PATTERN_CACHE.S.colval,
            colptrL = PATTERN_CACHE.S.colptr
        ) 
    end
    S = VecchiaMLE_Run(input)[2] 

    # calculating the scaling coefficient
    scale = mean(diag(S))^(-2) + mean(diag(R))
    scale *= n
    scale *= Ne # Do we add this as well?
 
    S ./= sqrt(scale)

    if PATTERN_CACHE.S === nothing
        cache_pattern!(S, :S, PATTERN_CACHE)
    end

    # Next form kalman filter
    K = L' \ (L \ H')
    K .= K * S * S'
    
    # Next perform update
    xf .+= K * inn
    return
end


