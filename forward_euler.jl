function forward_euler(u, N, dx, dy, dt, cx, cy, nu)
    # Computes the next time step using Forward Euler with Periodic BCs
    u = reshape(u, N, N);
    # Initialize new solution array
    u_new = u;
    
    # Compute derivatives using finite difference stencil (central for diffusion, upwind for advection)
    for i = 2:N - 1
        for j = 2:N - 1
            # Advection (Upwind)
            adv_x = -cx * (u[i, j] - u[i-1, j]) / dx;
            adv_y = -cy * (u[i, j] - u[i, j-1]) / dy;
    
            # Diffusion (Central Difference)
            diff_x = nu * (u[i+1, j] - 2 * u[i, j] + u[i-1, j]) / dx^2;
            diff_y = nu * (u[i, j+1] - 2 * u[i, j] + u[i, j-1]) / dy^2;
    
            # Forward Euler update
            u_new[i, j] = u[i, j] + dt * (adv_x + adv_y + diff_x + diff_y);
        end
    end
    
    # Apply Periodic Boundary Conditions
    u_new[1, :] = u_new[N-1, :]; # Left boundary wraps to N-1
    u_new[N, :] = u_new[2, :]; # Right boundary wraps to 2
    u_new[:, 1] = u_new[:, N-1]; # Bottom boundary wraps to N-1
    u_new[:, N] = u_new[:, 2]; # Top boundary wraps to 2
    return u_new
end