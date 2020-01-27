using ForwardDiff

function compute(; x, y)
    return @. 2.0*x + 3.0*y
end

function compute!(; x, y, z)
    @. z = 2.0*x + 3.0*y
    return nothing
end

function compute!(; inputs::AbstractDict, outputs::AbstractDict)
    args = merge(inputs, outputs)
    compute!(; args...)
    return nothing
end

function test1()

    y = [10.0]

    function f1(wrt)
        return compute(x=wrt, y=y)
    end
    @show f1([0.5])
    @show ForwardDiff.jacobian(f1, [0.5])

    inputs = Dict(:x => [0.5], :y => y)
    @show compute(; inputs...)

    function f2(wrt)
        inputs[:x] = wrt
        return compute(; inputs...)
    end
    @show f2([0.5])
    # @show ForwardDiff.jacobian(f2, [0.5])
    
    wrt_sym = :x
    wrt_val = pop!(inputs, wrt_sym)
    function f3(wrt)
        args = merge(inputs, Dict(wrt_sym => wrt))
        return compute(; args...)
    end
    @show f3([0.5])
    @show ForwardDiff.jacobian(f3, [0.5])
    inputs[wrt_sym] = wrt_val

end

function test2()
    y = [10.0]

    function f1!(ret, wrt)
        compute!(x=wrt, y=y, z=ret)
        return nothing
    end
    z = [0.0]
    f1!(z, [0.5])
    @show z
    z = [0.0]
    @show ForwardDiff.jacobian(f1!, z, [0.5])

    inputs = Dict(:x => [0.5], :y => y)
    outputs = Dict(:z => [0.0])
    args = merge(inputs, outputs)
    compute!(; args...)
    @show z

    function f2!(ret, wrt)
        inputs[:x] = wrt
        outputs[:z] = ret
        args = merge(inputs, outputs)
        compute!(; args...)
        return nothing
    end
    z = [0.0]
    f2!(z, [0.5])
    @show z
    z = [0.0]
    # @show ForwardDiff.jacobian(f2!, z, [0.5])

    wrt_sym = :x
    wrt_val = pop!(inputs, wrt_sym)
    ret_sym = :z
    ret_val = pop!(outputs, ret_sym)
    function f3!(ret, wrt)
        args = merge(inputs, Dict(wrt_sym => wrt), outputs, Dict(ret_sym => ret))
        compute!(; args...)
        return nothing
    end
    z = [0.0]
    f3!(z, [0.5])
    @show z
    z = [0.0]
    @show ForwardDiff.jacobian(f3!, z, [0.5])

end

function test3()
    inputs = Dict(:x => [0.5], :y => [10.0])
    outputs = Dict(:z => [0.0])

    of_sym = :z
    of_val = pop!(outputs, of_sym)
    wrt_sym = :x
    wrt_val = pop!(inputs, wrt_sym)
    function f1!(ret, wrt)
        args = merge(inputs, Dict(wrt_sym => wrt),
                     outputs, Dict(of_sym => ret))
        compute!(; args...)
        return nothing
    end

    z = [0.0]
    f1!(z, [0.5])
    @show z

end

function main()
    test1()
    test2()
    # test3()
end

main()
