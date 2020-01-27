using ForwardDiff

function compute!(; inputs::AbstractDict, outputs::AbstractDict)
    @. outputs[:z] = 2.0*inputs[:x] + 3.0*inputs[:y]
    return nothing
end

function test3()
    inputs = Dict(:x => [0.5], :y => [10.0])
    outputs = Dict(:z => [0.0])

    of_sym = :z
    of_val = pop!(outputs, of_sym)
    wrt_sym = :x
    wrt_val = pop!(inputs, wrt_sym)
    function f1!(wrt)
        local_inputs = merge(inputs, Dict(wrt_sym => wrt))
        local_outputs = merge(outputs, Dict(of_sym => similar(wrt)))
        @show local_inputs, local_outputs
        compute!(; inputs=local_inputs, outputs=local_outputs)
        return local_outputs[of_sym]
    end
    @show f1!([0.5])
    @show ForwardDiff.jacobian(f1!, [0.5])

end

function main()
    test3()
end

main()
