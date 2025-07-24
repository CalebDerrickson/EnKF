function BabyKF(xf, y, H, R, infl, rho, ptGrid::AbstractVector, observe_index::AbstractVector, strat::Strategy, k, PATTERN_CACHE)
    num_states, N = size(xf) # 2500 x 50 
    num_observation = size(y, 1)

    # Add noise to observation
    y_perturbed = repeat(y, 1, N) .+ sqrt.(R) * randn(num_observation, N)

    # Add inflation
    xfm = mean(xf, dims=2)
    xf_dev = (1 / sqrt(N-1)) * (xf .- repeat(xfm, 1, N))
    xf .= sqrt(N - 1) * sqrt(infl) * xf_dev + repeat(xfm, 1, N)
    
    # Using inn as the observed update of kalman filter
    inn = y_perturbed .- view(xf, observe_index, :)

    if strat == localization 
        Localization(xf, y, xf_dev, inn, observe_index, rho, R)
    elseif strat == OneVecchia
        OneVecchiaEnKF(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k, PATTERN_CACHE)
    elseif strat == TwoVecchia
        TwoVecchiaEnKF(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k, PATTERN_CACHE)
    elseif strat == Empirical
        EmpiricalAnalysis(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k)
    end
end

function Localization(xf, y, xf_dev, inn, observe_index, rho, R)
    num_states, N = size(xf) # 2500 x 50 
    num_observation = size(y, 1)
    rhoPH = view(rho, :, observe_index)
    rhoHPHt = view(rho, observe_index, observe_index)

    zb = view(xf, observe_index, :)
    zbm = mean(zb, dims=2) 
    Zb = (1 / sqrt(N - 1)) * (zb .- repeat(zbm, 1, N))
    temp = LinearAlgebra.Symmetric((Zb * Zb') .+ R)

    K = (rhoHPHt .* temp) \ inn
    xf .+= rhoPH .* (xf_dev * Zb') * K
    return nothing
end

function EmpiricalAnalysis(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k)
    num_states, N = size(xf)

    zb = xf[observe_index, :]
    zbm = mean(zb, dims=2) 
    Zb = (1 / sqrt(N - 1)) * (zb .- repeat(zbm, 1, N))

    K = ( (Zb * Zb') .+ R) \ inn
    xf .+= (xf_dev * Zb') * K
    return nothing
end

function OneVecchiaEnKF(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k, PATTERN_CACHE)
    num_states, Ne = size(xf)
    num_observation = size(y, 1)

    # Have to transpose them since that's how VecciaMLE parses them. 
    xf_mat = Matrix{Float64}(xf')
    xfm = mean(xf_mat, dims = 1) # mean by col. 
    xf_mat .-= repeat(xfm, Ne, 1)

    n = num_states
    if PATTERN_CACHE.L === nothing
        input = VecchiaMLEInput(n, k, xf_mat, Ne, 5, 1; ptGrid=ptGrid, skip_check=true) #sparsityGeneration=VecchiaMLE.HNSW)
    else
        input = VecchiaMLEInput(n, k, xf_mat, Ne, 5, 1; ptGrid=ptGrid, skip_check=true,
            rowsL = PATTERN_CACHE.L.rowval,
            colsL = PATTERN_CACHE.L.colval,
            colptrL = PATTERN_CACHE.L.colptr,
            #sparsityGeneration = VecchiaMLE.HNSW
        )
    end
    d, L = VecchiaMLE_Run(input)
    VecchiaMLE.print_diagnostics(d)

    if PATTERN_CACHE.L === nothing
        cache_pattern!(L, :L, PATTERN_CACHE)
    end


    # The below code is Phil's idea (SMW)
    # K = Rₖ⁻²[R - Z( (N-1)I + ZᵀRₖ⁻¹Z )⁻¹Zᵀ]
    # Z = HₖX

    Z = view(xf_mat, :, observe_index)
    K = (L' \ (L \ H')) * (R^2 \ (R .- (Z' * ((Z * (R \ Z') + (Ne-1)*I) \ Z))))
    K ./= (Ne-1)

    # Next perform update
    xf .+= (1 / (Ne-1)) .* K * inn
    
    
    return
end


function TwoVecchiaEnKF(xf, y, xf_dev, inn, observe_index, rho, R, ptGrid, H, k, PATTERN_CACHE)
    num_states, Ne = size(xf)
    num_observation = size(y, 1)

    # Have to transpose them since that's how VecciaMLE parses them. 
    xf_mat = Matrix{Float64}(xf')
    xfm = mean(xf_mat, dims = 1) # mean by col. 
    xf_mat .-= repeat(xfm, Ne, 1)
    
    n = num_states
    
    if PATTERN_CACHE.L === nothing
        input = VecchiaMLEInput(n, k, xf_mat, Ne, 5, 1; ptGrid=ptGrid, skip_check=true)
    else
        input = VecchiaMLEInput(n, k, xf_mat, Ne, 5, 1; ptGrid=ptGrid, skip_check=true,
            rowsL = PATTERN_CACHE.L.rowval,
            colsL = PATTERN_CACHE.L.colval,
            colptrL = PATTERN_CACHE.L.colptr
        )
    end

    L = VecchiaMLE_Run(input)[2]

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
    n = length(observe_index)

    samples =  randn(Ne, Ne) * chol

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


