using ForwardDiff

struct SquareIt
    a
end

function compute!(self::SquareIt; x, y, z1, z2)
    a = self.a
    @. z1 = a*x*x + y*y
    @. z2 = a*x + y
end

function compute!(self::SquareIt, inputs::AbstractDict, outputs::AbstractDict)
    args = merge(inputs, outputs)
    compute!(self; args...)
end

function compute_partials!(self::SquareIt, inputs::AbstractDict, partials::AbstractDict)

    # Need to create an outputs dictionary. Should be same type as inputs, I
    # guess.
    outputs = typeof(inputs)()
    @show outputs
    outputs_type = eltype(values(outputs))
    for ((of, wrt), deriv) in partials
        # TODO: figure out how to get each output's size.
        size = (1,)
        outputs[of] = outputs_type(undef, size)
    end
    @show outputs

    for ((of, wrt), deriv) in partials
        @show of, wrt, deriv
        of_val = pop!(outputs, of)
        @show of, of_val
        function f(x)
            args = merge(inputs, outputs, Dict(of => of_val))
            args[wrt] = x
            @show args
            compute!(self; args...)
            return args[of]
        end
        @show f(inputs[wrt])
        @show ForwardDiff.jacobian(f, inputs[wrt])
        outputs[of] = of_val
    end
end

comp = SquareIt(8)
the_inputs = Dict(:x => [2.0], :y => [3.0])
the_outputs = Dict(:z1 => [0.0], :z2 => [0.0])
@show the_inputs, the_outputs
@show compute!(comp, the_inputs, the_outputs)

the_partials = Dict((:z1, :x) => [0.0],
                    (:z1, :y) => [0.0],
                    (:z2, :x) => [0.0],
                    (:z2, :y) => [0.0])

compute_partials!(comp, the_inputs, the_partials)
