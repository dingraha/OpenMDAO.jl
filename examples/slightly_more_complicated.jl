using ForwardDiff

function compute!(; x, y, z)
    @. z = 2.0*x + 3.0*y
    return nothing
end

function compute!(; inputs::AbstractDict, outputs::AbstractDict)
    args = merge(inputs, outputs)
    compute!(; args...)
    return nothing
end

function test3()
    inputs = Dict(:x => [0.5], :y => [10.0])
    outputs = Dict(:z => [0.0])

    of_sym = :z
    of_val = pop!(outputs, of_sym)
    wrt_sym = :x
    wrt_val = pop!(inputs, wrt_sym)
    function f1!(ret, wrt)
        local_inputs = merge(inputs, Dict(wrt_sym => wrt))
        local_outputs = merge(outputs, Dict(of_sym => ret))
        compute!(; inputs=local_inputs, outputs=local_outputs)
        return nothing
    end

    z = [0.0]
    f1!(z, [0.5])
    @show z

end

function main()
    test3()
end

main()
