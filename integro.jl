function integro(u, N, dt, kernel)

    u = reshape(u, N*N, 1)
    # Define the nonlinearity f(N)
    r = 0.2
    K = 1.0
    f(s) = s .* (1 + r .- (r / K) .* s) # (1+r)s - (r / K) s^2
    return u .+ dt .* (kernel * f(u))

end