function BabyKF(xf, y, H, R, infl, rho, observe_index::AbstractVector, localize=false)
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


        println("Zb size: ", size(Zb))
        println("R size: ", size(R))
        println("inn size: ", size(inn))
        println("K size: ", size(K))
        println("H size: ", size(H))

    else
        verbose=false
        # Implement VecchiaMLE here. H is the linear mapping from state space to observation locations (xyGrid -> ptGrid). 
        #n = cld(sqrt(N)), k = 10, samples = ensemble (xf), number_of_samples, ptGrid  
        xf_zeroed = Matrix{Float64}((xf .- mean(xf, dims=2))')
        n = Int(sqrt(size(xf_zeroed, 2)))
        Number_of_Samples = size(xf_zeroed, 1)


        ptGrid = VecchiaMLE.generate_safe_xyGrid(n)
        input = VecchiaMLEInput(n, 10, xf_zeroed, Number_of_Samples, 5, 1; ptGrid=ptGrid)
        _, L = VecchiaMLE_Run(input)
        L_inv = inv(L)
        diag_R = view(R, diagind(R))


        temp = zeros(N^2, N^2)
        fill!(view(temp, observe_index, observe_index), 1.0) 
        view(temp, observe_index, observe_index) ./= diag_R

        # Inner part of the Kalman filter product
        # I.e., ( Rₖ - Hₖ(Bₖ⁻¹ + HₖᵀRₖ⁻¹Hₖ)⁻¹Hₖᵀ )
        # The inv() is really ugly, and needs to be changed.
        inner = view(inv(L*L' + temp), observe_index, observe_index)
        inner .= R - inner
        
        # outer part of Kalman filter product
        # I.e., BₖHₖᵀRₖ⁻¹(inner)Rₖ⁻¹
        inner ./= R
        inner ./= R # twice from both sides?

        # Now need to map from 100 x 100 to 2500 x 100. 
        # is there a better way?
        outer = view(temp, :, observe_index)
        fill!(outer, 0.0)

        println("inner size: ", size(inner))
        println("N: ", N)
        println("outer size: ", size(outer))
        println("observe_index : ", size(observe_index))
        view(outer, observe_index, :) .= inner

        lmul!(L_inv, outer)
        lmul!(l_inv', outer)
        K = outer


        # now to apply the Kalman filter
        verbose = true
        verbose && println("xf_zeroed size: ", size(xf_zeroed))
        verbose && println("K size: ", size(K))
        verbose && println("xf_dev size: ", size(xf_dev))
        verbose && println("xf size: ", size(xf))

        t1 = xf_zeroed * K
        t2 = xf_dev * t1
        xa = xf + t2


        verbose && println("ptGrid size: ", size(ptGrid))
        verbose && println("N: ", N)
        
        verbose && println("R size: ", size(R))
        verbose && println("inn size: ", size(inn))
        verbose && println("L size: ", size(L))
        verbose && println("\n")
        verbose && println("t1 size: ", size(t1))
        verbose && println("t2 size: ", size(t2))
    end
    

    return xa
end

