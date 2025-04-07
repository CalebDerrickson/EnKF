function BabyKF(xf, y, H, R, infl, rho, observe_index, localize=false)
    num_states, N = size(xf)
    num_observation = size(y, 1)

    # Since Julia 0.7 sqrt of a matrix is sqrtm in Matlab
    y_perturbed = repeat(y, 1, N) + sqrt.(R) * randn(num_observation, N)

    xfm = mean(xf, dims=2)
    xf_dev = (1 / sqrt(N-1)) * (xf - repeat(xfm, 1, N))
    xf = sqrt(N - 1) * infl * xf_dev + repeat(xfm, 1, N)

    observable = H * xf
    inn = y_perturbed - observable
    
    if localize
        rhoPH = rho[:, observe_index]
        rhoHPHt = rho[observe_index, observe_index]

        zb = observable
        zbm = mean(zb, dims=2) 
        Zb = (1 / sqrt(N - 1)) * (zb .- repeat(zbm, 1, N))

        K = (rhoHPHt .* (Zb * Zb') + R) \ inn
        xa = xf + rhoPH .* (xf_dev * Zb') * K

    else
        # Implement VecchiaMLE here. H is the linear mapping from state space to observation locations (xyGrid -> ptGrid). 
        #n = cld(sqrt(N)), k = 10, samples = ensemble (xf), number_of_samples, ptGrid  
        zb = Matrix{Float64}((observable .- mean(observable, dims=2))')
        n = Int(sqrt(size(zb, 2)))
        Number_of_Samples = size(zb, 1)


        ptGrid = VecchiaMLE.generate_safe_xyGrid(n)
        println("ptGrid size: ", size(ptGrid))
        
        println("N: ", N)
        println("zb size: ", size(zb))
        
        input = VecchiaMLEInput(n, 10, zb, Number_of_Samples, 5, 1; ptGrid=ptGrid)
        _, L = VecchiaMLE_Run(input)

        L_inv = inv(L)

        println("R size: ", size(R))
        println("inn size: ", size(inn))
        println("L size: ", size(L))
        println("xf_dev size: ", size(xf_dev))

        pt1 = L_inv' * L_inv
        pt2 = pt1 + R
        K = pt2 \ inn 
        println("K size: ", size(K))
        println("\n")

        t1 = xf_dev
        t2 = L_inv' * K
        println("t1 size: ", size(t1))
        println("t2 size: ", size(t2))


        xa = xf + t1 * t2
    end
    

    return xa
end

