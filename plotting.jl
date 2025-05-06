using Plots
gr()

function main()

    max_iter = 25
    len = 100
    ks = 1:10
    val = 4
    input = readdlm("EnKF_output_4444.txt", ',')
    
    p1 = plotVecchia1(max_iter, len, ks, input, val)
    savefig(p1, "vecchia1"* (val == 4 ? "_rms" : "_kl") * ".png")

    p2 = plotVecchia2(max_iter, len, ks, input, val)
    savefig(p2, "vecchia2"* (val == 4 ? "_rms" : "_kl") * ".png")
    
end


function plotVecchia1(max_iter, len, ks, input, val)

    lines = zeros(length(ks)+1, len)
    titles = ["localize"]
    carry = len
    
    lines[1, :] .= input[1:carry, val]
    for i in ks
        lines[i+1, :] .= input[(1:len).+carry, val]
        carry += len
        push!(titles, "k = $(i)")
    end
    p1 = Plots.plot()
    i = 1
    for line in eachrow(lines)
        if i == 1
            Plots.plot!(p1, 1:len, line, label=titles[i], lc=:black)
        else
            Plots.plot!(p1, 1:len, line, label=titles[i])
        end
        i+=1
    end
    plot!(yscale=:log10, minorgrid=true)
    #ylims!(1900, 5e7)
    #title!(p1, "max_iter: $(max_iter)")
    ylabel!(p1, val == 4 ? "RMSE" : "KL")
    xlabel!(p1, "Time step")
    title!(p1, val == 4 ? "RMS 1VecchiaMLE" : "KL 1VechiaMLE")
    plot!(p1, legend=:outerbottom, legendcolumns=5)

    return p1
end

function plotVecchia2(max_iter, len, ks, input, val)
    lines = zeros(length(ks)+1, len)
    titles = ["localize"]
    carry = len
    
    lines[1, :] .= input[1:carry, val]
    carry = (length(ks)+1)*len
    for i in ks
        lines[i+1, :] .= input[(1:len).+carry, val]
        carry += len
        push!(titles, "k = $(i)")
    end
    p2 = Plots.plot()
    i = 1
    for line in eachrow(lines)
        if i == 1
            Plots.plot!(p2, 1:len, line, label=titles[i], lc=:black)
        else
            Plots.plot!(p2, 1:len, line, label=titles[i])
        end
        i+=1
    end
    plot!(p2, yscale=:log10, minorgrid=true)
    #ylims!(0.004, 0.01)
    title!(p2, "max_iter: $(max_iter)")
    ylabel!(p2, val == 4 ? "RMSE" : "KL")
    xlabel!(p2, "Time step")
    title!(p2, val == 4 ? "RMS 2VecchiaMLE" : "KL 2VechiaMLE")
    plot!(p2, legend=:outerbottom, legendcolumns=5)

    return p2
end