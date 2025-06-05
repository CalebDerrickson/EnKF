function integro(u, N, dt, kernel)

    u = reshape(u, N*N, 1)
    # Define the nonlinearity f(N)
    r = 0.0
    K = 1.0
    f(s) = r .* s .* (1 .- s./ K)


    #kernel .*= dx * dy
    return u .+ dt .* (kernel * f(u))

end