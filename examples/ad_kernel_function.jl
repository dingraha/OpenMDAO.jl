using ForwardDiff

struct SquareIt
    a
end

function compute!(self::SquareIt; x, y, z1, z2)
    @. z1 = 2.0*self.a*x + 3.0*y
    @. z2 = 3.0*self.a*x + 4.0*y
    return nothing
end

function compute!(self::SquareIt, inputs::AbstractDict, outputs::AbstractDict)
    args = merge(inputs, outputs)
    compute!(self; args...)
    return nothing
end

function compute_partials!(self, inputs::AbstractDict, partials::AbstractDict)

    # Need to create an outputs dictionary. Should be same type as inputs, I
    # guess.
    outputs = typeof(inputs)()
    outputs_type = eltype(values(outputs))
    for ((of_sym, wrt_sym), deriv) in partials
        # TODO: figure out how to get each output's size.
        size = (1,)
        outputs[of_sym] = outputs_type(undef, size)
    end

    for ((of_sym, wrt_sym), deriv) in partials
        @show of_sym, wrt_sym
        of_val = pop!(outputs, of_sym)
        wrt_val = pop!(inputs, wrt_sym)

        function f(x)
            local_inputs = merge(inputs, Dict(wrt_sym => x))
            local_outputs = merge(outputs, Dict(of_sym => similar(x)))
            compute!(self, local_inputs, local_outputs)
            return local_outputs[of_sym]
        end
        deriv .= ForwardDiff.jacobian(f, wrt_val)

        outputs[of_sym] = of_val
        inputs[wrt_sym] = wrt_val
    end

    return nothing
end

function main()
    T = Float64
    comp = SquareIt(-1.0)
    inputs = Dict(:x => T[2.0], :y => T[3.0])
    outputs = Dict(:z1 => T[0.0], :z2 => T[0.0])

    partials = Dict((:z1, :x) => zeros(T, 1, 1),
                    (:z1, :y) => zeros(T, 1, 1),
                    (:z2, :x) => zeros(T, 1, 1),
                    (:z2, :y) => zeros(T, 1, 1))

    compute!(comp, inputs, outputs)
    @show inputs
    @show outputs
    compute_partials!(comp, inputs, partials)
    @show partials
end

main()
