using Plots
using DelimitedFiles

function main()
    seed = 4681
    dts = [1, 2, 4, 5].*0.001
    
    for dt in dts
        filename = "EnKF_output_$(seed)_$(dt).txt"
        input = readdlm(filename)  
        p1, p2 = plot(input, dt)
        savefig(p1, "Vecchia1_dt_$(dt).png")
        savefig(p2, "Vecchia2_dt_$(dt).png")
    end
    
end

function plot(input, dt)

    # Check for zeros
    for i in 1:size(input, 1)
        input[i, input[i, :] .==0] .= maximum(input[i, :])
    end

    i = 1
    titles = ["localize"]
    for line in 1:10
        push!(titles, "k = $(i)")
        i+=1
    end
    i = 1
    p1 = Plots.plot()
    for line in 1:11
        if i == 1
            plot!(p1, dt:dt:size(input, 2)*dt, input[line, :], label=titles[i], lc=:black, linewitdth=2)
        else
            plot!(p1, dt:dt:size(input, 2)*dt, input[line, :], label=titles[i], linewidth=0.75)
        end
        i+=1
    end
    i = 1
    p2 = Plots.plot()
    for line in 1:11
        if i == 1
            plot!(p2, dt:dt:size(input, 2)*dt, input[line, :], label=titles[i], lc=:black, linewidth=2)
        else
            plot!(p2, dt:dt:size(input, 2)*dt, input[line+10, :], label=titles[i], linewidth=0.75)
        end
        i+=1
    end

    plot!(p1, yscale=:log10, minorgrid=true)
    ylims!(p1, 0.001, 0.1)
    #title!(p1, "max_iter: $(max_iter)")
    ylabel!(p1, "RMSE")
    xlabel!(p1, "Time step")
    title!(p1, "RMS 1VecchiaMLE - dt = $(dt)")
    plot!(p1, legend=:outerbottom, legendcolumns=5, dpi=1000)

    plot!(p2, yscale=:log10, minorgrid=true)
    ylims!(p2, 0.001, 1)
    ylabel!(p2, "RMSE")
    xlabel!(p2, "Time step")
    title!(p2, "RMS 2VecchiaMLE - dt = $(dt)")
    plot!(p2, legend=:outerbottom, legendcolumns=5, dpi=1000)

    return [p1, p2]
end