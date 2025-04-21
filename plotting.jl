using Plots

function main()
    max_iter = 25
    len = 100
    ks = 1:7
    lines = zeros(length(ks)+1, len)
    input = readdlm("EnKF_output_max_iter_$(max_iter).txt", ',')
    titles = ["localize"]
    carry = 1
    

    for i in ks
        lines[i, :] .= input[carry:i*len, 3]
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
    ylims!(1e-3, 10)
    title!("max_iter: $(max_iter)")
    plot!(legend=:outerbottom, legendcolumns=5)
    display(p)
end