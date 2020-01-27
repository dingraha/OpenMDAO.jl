using ForwardDiff

struct SquareIt
    a
    ss_sizes
    ss_inputs
    ss_outputs
end
SquareIt(; a, ss_sizes, ss_inputs, ss_outputs) = SquareIt(a, ss_sizes, ss_inputs, ss_outputs)

function compute!(self::SquareIt; x, y, z1, z2)
    z1 .= 2 .* self.a .* sum(x, dims=1) .+ y
    r = reshape(1:self.ss_sizes[:k], 1, self.ss_sizes[:k], 1, 1)
    z2 .= 3 .* sum(x, dims=1) .+ r .- y
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
    num_nodes = 2
    num_blades = 3
    num_radial = 4
    n_d = 3
    ss_sizes = Dict(:l => n_d, :k => num_radial, :j => num_blades, :i => num_nodes, :s =>1)
    ss_inputs = Dict(:x => [:l, :s, :j, :i], :y => [:s, :s, :j, :s])
    ss_outputs = Dict(:z1 => [:s, :s, :j, :i], :z2 => [:s, :k, :j, :i])

    comp = SquareIt(a=-1.0,
                    ss_sizes=ss_sizes,
                    ss_inputs=ss_inputs,
                    ss_outputs=ss_outputs)
    inputs = Dict(
        :x => fill(2.0, [ss_sizes[s] for s in ss_inputs[:x]]...),
        :y => fill(3.0, [ss_sizes[s] for s in ss_inputs[:y]]...))
    outputs = Dict(
        :z1 => fill(0.0, [ss_sizes[s] for s in ss_outputs[:z1]]...),
        :z2 => fill(0.0, [ss_sizes[s] for s in ss_outputs[:z2]]...))

    # First step: create a partials dictionary. How do I do that?
    partials = Dict{Tuple{keytype(outputs), keytype(inputs)}, valtype(inputs)}()
    # @show partials
    # ss_inputs_all = Set{Symbol}()
    # ss_outputs = Set{Symbol}()
    # for (osym, oss) in ss_outputs
    #     for (isym, iss) in ss_inputs
    #         # pss = cat(setdiff(oss, [:s]), setdiff(iss, cat(oss, [:s], dims=1)), dims=1)
    #         # @show oss, iss, pss
    #     end
    # end

    ss_inputs_all = union(Set.(values(ss_inputs))...)
    ss_outputs_all = union(Set.(values(ss_outputs))...)
    ss_outputs_common = intersect(Set.(values(ss_outputs))...)
    @show ss_inputs_all
    @show ss_outputs_all
    @show ss_outputs_common
    ss_inputs_not_in_outputs = setdiff(ss_inputs_all, ss_outputs_all)
    @show ss_inputs_not_in_outputs

    # compute!(comp, inputs, outputs)
    # @show inputs
    # @show outputs
    # compute_partials!(comp, inputs, partials)
    # @show partials
end

main()
