using Plots

function main()
    max_iter = 25
    len = 100
    ks = 1:10
    lines = zeros(length(ks)+1, len)
    input = readdlm("EnKF_output.txt", ',')
    titles = ["localize"]
    carry = 101
    
    lines[1, :] .= input[1:100, 3]
    for i in ks
        lines[i+1, :] .= input[carry:(i+1)*len, 3]
        carry += len
        push!(titles, "k = $(i)")
    end
    p = Plots.plot()
    i = 1
    for line in eachrow(lines)
        if i == 1
            Plots.plot!(p, 1:len, line, label=titles[i], lc=:black)
        else
            Plots.plot!(p, 1:len, line, label=titles[i])
        end
        i+=1
    end
    plot!(yscale=:log10, minorgrid=true)
    #ylims!(0.004, 0.01)
    #title!("max_iter: $(max_iter)")
    ylabel!("RMSE")
    xlabel!("Time step")
    title!("EnKF - Localization to VecchiaMLE")
    plot!(legend=:outerbottom, legendcolumns=5)

    display(p)
end