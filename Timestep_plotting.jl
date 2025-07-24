using Plots
using DelimitedFiles

function main()
    seed = 7763
    dts = [8].*0.001
    ks = 8:10
    N = 196

    for dt in dts
        filename = "EnKF_output_$(seed)_$(dt).txt"
        input = readdlm(filename)  
        p2 = plot(input, dt, ks, N)
        #savefig(p1, "Vecchia1_dt_$(dt).png")
        savefig(p2, "Vecchia2_dt_$(dt).png")
    end
    
end

function plot(input, dt, ks, N)

    # Check for zeros
    for i in 1:size(input, 1)
        input[i, input[i, :] .==0] .= maximum(input[i, :])
    end

    titles = ["localize"]
    for line in ks
        push!(titles, "k = $(line)")
    end

    i = 1
    p2 = Plots.plot()
    for line in 1:(length(ks)+1)
        if i == 1
            plot!(p2, dt:dt:size(input, 2)*dt, input[line, :], label=titles[i], lc=:black, linewidth=2)
        else
            plot!(p2, dt:dt:size(input, 2)*dt, input[line, :], label=titles[i], linewidth=0.75)
        end

        i+=1
    end
    plot!(p2, minorgrid=true)
    #ylims!(p2, 0.001, 1)
    ylabel!(p2, "RMSE")
    xlabel!(p2, "Time step")
    title!(p2, "RMS 2VecchiaMLE - dt = $(dt), N = $(N)")
    plot!(p2, legend=:outerbottom, legendcolumns=5, dpi=1000)

    return p2
end