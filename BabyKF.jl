function BabyKF(xf, y, H, R, infl, rho, ptGrid::AbstractVector, observe_index::AbstractVector, localize::Bool, k, rep, L, iteration)
    num_states, N = size(xf) # 2500 x 50 
    num_observation = size(y, 1)

    # Add noise to observation
    y_perturbed = repeat(y, 1, N) + sqrt.(R) * (randn(num_observation, N))

    # Add inflation
    xfm = mean(xf, dims=1)
    xf_dev = (1 / sqrt(N-1)) * (xf .- repeat(xfm, num_states, 1))
    xf .= sqrt(N - 1) * sqrt(infl) * xf_dev + repeat(xfm, num_states, 1)
    
    # Using inn as the observed update of kalman filter
    inn = y_perturbed .- xf[observe_index, :]

    if localize
        rhoPH = rho[:, observe_index]
        rhoHPHt = rho[observe_index, observe_index]

        zb = xf[observe_index, :]
        zbm = mean(zb, dims=2) 
        Zb = (1 / sqrt(N - 1)) * (zb .- repeat(zbm, 1, N))

        K = (rhoHPHt .* (Zb * Zb') .+ R) \ inn
        xf .+= rhoPH .* (xf_dev * Zb') * K

        return

    else
        # Have to transpose them since that's how VecciaMLE parses them. 
        xf_mat = Matrix{Float64}(xf')
        xfm = mean(xf_mat, dims = 2)
        #xf_mat .-= repeat(xfm, 1, num_states)
        
        n = Int(sqrt(size(xf_mat, 2)))
        input = VecchiaMLEInput(n, k, xf_mat, N, 5, 1; ptGrid=ptGrid)
        
        zb = xf
        zbm = mean(zb, dims=2) 
        Zb = (1 / sqrt(N - 1)) * (zb .- repeat(zbm, 1, N))
        ACov = Zb*Zb'

        d, L = VecchiaMLE_Run(input)
        
        
        
        println(findnz(sparse(L[k, :]))[2])
        println("grad: ", d.normed_grad_value)
        println("cons: ", d.normed_constraint_value)
        println("KLDiv: ", VecchiaMLE.KLDivergence(ACov, L))


        rep .= L' \ (L \ H')
        xf .+= rep * ((rep[observe_index, :].+R) \ inn)
        return
    end
    
    
end

