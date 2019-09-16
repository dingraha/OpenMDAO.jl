using ForwardDiff

# struct SquareIt
#     a
#     ss_sizes
#     ss_inputs
#     ss_outputs
# end
# SquareIt(; a, ss_sizes, ss_inputs, ss_outputs) = SquareIt(a, ss_sizes, ss_inputs, ss_outputs)

# function compute!(self::SquareIt; x, y, z1, z2)
#     z1 .= 2 .* self.a .* sum(x, dims=1) .+ y
#     r = reshape(1:self.ss_sizes[:k], 1, self.ss_sizes[:k], 1, 1)
#     z2 .= 3 .* sum(x, dims=1) .+ r .- y
#     return nothing
# end

# function compute!(self::SquareIt, inputs::AbstractDict, outputs::AbstractDict)
#     args = merge(inputs, outputs)
#     compute!(self; args...)
#     return nothing
# end

function compute!(; x, y, z1, z2)
    z1 .= 2.0 .* sum(x, dims=1) .+ 3.0 .* y
    num_radial = size(z2, 2)
    r = reshape(1:num_radial, 1, num_radial, 1, 1)
    z2 .= 3.0 .* sum(x, dims=1) .+ r .- y
    return nothing
end

# function compute!(; inputs::AbstractDict, outputs::AbstractDict)
#     args = merge(inputs, outputs)
#     compute!(; args...)
#     return nothing
# end

function get_dict_of_array_offsets(var_ss, ss_sizes, ss_outputs_common)
    s, e = 1, 1
    d = Dict{Symbol, Tuple{Int64, Int64}}()
    for (var, subscripts) in var_ss
        len = 1
        for ss in subscripts
            if ! (ss in ss_outputs_common)
                len *= ss_sizes[ss]
            end
        end
        e += len - 1
        d[var] = (s, e)
        s = e + 1
        e = s
    end
    return e-1, d
end

function get_dict_of_array_sizes(var_ss, ss_sizes, ss_outputs_common)
    d = Dict{Symbol, Array{Int64, 1}}()
    for (var, subscripts) in var_ss
        varsize = Int64[]
        for ss in subscripts
            if ss in ss_outputs_common
                push!(varsize, 1)
            else
                push!(varsize, ss_sizes[ss])
            end
        end
        d[var] = varsize
    end
    return d
end

function init_jacobian(ss_inputs, ss_outputs, ss_outputs_unique, ss_inputs_unique, ss_outputs_common, ss_sizes)
    n_ss = length(ss_sizes) - 1  # assume that ss_sizes has a dummy index.
    deriv_size = ones(Int64, n_ss)
    deriv_size[end-length(ss_outputs_common)+1:end] = [ss_sizes[ss] for ss in ss_outputs_common]
    d = Dict{Tuple{Symbol, Symbol}, Array{Float64, n_ss}}()
    for (of_var, of_subscripts) in ss_outputs
        for (wrt_var, wrt_subscripts) in ss_inputs
            idx = 1
            for ss in ss_outputs_unique
                if ss in ss_outputs[of_var]
                    deriv_size[idx] = ss_sizes[ss]
                end
                idx += 1
            end
            for ss in ss_inputs_unique
                if ss in ss_inputs[wrt_var]
                    deriv_size[idx] = ss_sizes[ss]
                end
                idx += 1
            end
            println("d[$of_var, $wrt_var] size = $(deriv_size)")
            d[of_var, wrt_var] = zeros(Float64, deriv_size...)
        end
    end
    return d
end

function main()
    num_nodes = 2
    num_blades = 3
    num_radial = 4
    n_d = 3
    ss_sizes = Dict(:l => n_d, :k => num_radial, :j => num_blades, :i => num_nodes, :s =>1)
    ss_inputs = Dict(:x => [:l, :s, :j, :i], :y => [:s, :s, :j, :s])
    ss_outputs = Dict(:z1 => [:s, :s, :j, :i], :z2 => [:s, :k, :j, :i])
    @show ss_sizes
    @show ss_inputs
    @show ss_outputs

    inputs = Dict(
        :x => fill(2.0, [ss_sizes[s] for s in ss_inputs[:x]]...),
        :y => fill(3.0, [ss_sizes[s] for s in ss_inputs[:y]]...))
    outputs = Dict(
        :z1 => fill(0.0, [ss_sizes[s] for s in ss_outputs[:z1]]...),
        :z2 => fill(0.0, [ss_sizes[s] for s in ss_outputs[:z2]]...))

    # I think I need to get away from using Sets.
    ss_outputs_common = intersect(values(ss_outputs)...)
    filter!(x->!=(x, :s), ss_outputs_common)
    @show ss_outputs_common

    ss_inputs_all = union(values(ss_inputs)...)
    filter!(x->!=(x, :s), ss_inputs_all)

    ss_outputs_all = union(values(ss_outputs)...)
    filter!(x->!=(x, :s), ss_outputs_all)

    # ss_inputs_unique = copy(ss_inputs_all)
    # ss_outputs_unique = copy(ss_outputs_all)
    # for ss in ss_outputs_common
    #     delete!(ss_inputs_unique, ss)
    #     delete!(ss_outputs_unique, ss)
    # end
    ss_inputs_unique = filter(x->!(x in ss_outputs_common), ss_inputs_all)
    ss_outputs_unique = filter(x->!(x in ss_outputs_common), ss_outputs_all)
    @show ss_inputs_unique
    @show ss_outputs_unique

    n_ss = length(ss_sizes) - 1
    # ss_outputs_common = collect(ss_outputs_common)
    ss_outputs_common_position = Dict(ss => findfirst(isequal(ss), ss_outputs_common) for ss in ss_outputs_common)
    @show ss_outputs_common_position

    ss_inputs_positions = Dict{Symbol, Dict{Symbol, Int}}()
    for (sym, subscripts) in ss_inputs
        ss_inputs_positions[sym] = Dict{Symbol, Int}()
        for ss in subscripts
            if ss != :s
                ss_inputs_positions[sym][ss] = findfirst(isequal(ss), subscripts)
            end
        end
    end
    @show ss_inputs_positions

    ss_outputs_positions = Dict{Symbol, Dict{Symbol, Int}}()
    for (sym, subscripts) in ss_outputs
        ss_outputs_positions[sym] = Dict{Symbol, Int}()
        for ss in subscripts
            if ss != :s
                ss_outputs_positions[sym][ss] = findfirst(isequal(ss), subscripts)
            end
        end
    end
    @show ss_outputs_positions

    slice_inputs = Dict{Symbol, Array{Union{Colon, Int64}, 1}}()
    for (sym, positions) in ss_inputs_positions
        slice = Array{Union{Colon, Int64}, 1}(undef, n_ss)
        fill!(slice, Colon())
        slice_inputs[sym] = slice
    end

    slice_outputs = Dict{Symbol, Array{Union{Colon, Int64}, 1}}()
    for (sym, positions) in ss_outputs_positions
        slice = Array{Union{Colon, Int64}, 1}(undef, n_ss)
        fill!(slice, Colon())
        slice_outputs[sym] = slice
    end

    (input_length, input_offsets) = get_dict_of_array_offsets(ss_inputs, ss_sizes, ss_outputs_common)
    @show input_length, input_offsets

    input_sizes = get_dict_of_array_sizes(ss_inputs, ss_sizes, ss_outputs_common)
    @show input_sizes

    (output_length, output_offsets) = get_dict_of_array_offsets(ss_outputs, ss_sizes, ss_outputs_common)
    @show output_length, output_offsets

    output_sizes = get_dict_of_array_sizes(ss_outputs, ss_sizes, ss_outputs_common)
    @show output_sizes

    inputs_packed = Array{Float64, 1}(undef, input_length)
    outputs_packed = Array{Float64, 1}(undef, output_length)

    function compute_diffable!(outputs::AbstractArray{T, 1}, inputs::AbstractArray{T, 1}) where {T}
        # Unpack the inputs.
        inputs_dict = Dict(var => reshape(inputs[s:e], input_sizes[var]...) for (var, (s, e)) in input_offsets)

        # Unpack the outputs.
        outputs_dict = Dict(var => reshape(outputs[s:e], output_sizes[var]...) for (var, (s, e)) in output_offsets)

        # Do it!
        args = merge(inputs_dict, outputs_dict)
        compute!(; args...)

        for (var, data) in outputs_dict
            (s, e) = output_offsets[var]
            outputs[s:e] = data[:]
        end

        return nothing
    end
    partials = init_jacobian(ss_inputs, ss_outputs, ss_outputs_unique, ss_inputs_unique, ss_outputs_common, ss_sizes)

    i = Dict{Symbol, Int64}()
    for I in CartesianIndices(Tuple(ss_sizes[ss] for ss in ss_outputs_common))
        for (k, v) in ss_outputs_common_position
            i[k] = I[v]
        end
        @show I, i

        for (sym, positions) in ss_inputs_positions
            for (ss, idx) in i
                if haskey(ss_inputs_positions[sym], ss)
                    pos = ss_inputs_positions[sym][ss]
                    slice_inputs[sym][pos] = idx
                end
            end
        end

        for (sym, positions) in ss_outputs_positions
            for (ss, idx) in i
                if haskey(ss_outputs_positions[sym], ss)
                    pos = ss_outputs_positions[sym][ss]
                    slice_outputs[sym][pos] = idx
                end
            end
        end

        # Now I need to pack all the inputs into a single array. How do I do
        # that? What I'd need is the size of the arrays that will eventually be
        # passed to the compute function. So that's each array's size with the
        # ss_outputs_common removed. I think I should be able to do that. Then I
        # need to get the offsets needed to stuff the inputs into that array.
        # Have that. Now I need to copy the arrays into the packed single
        # arrays.
        for (var, (s, e)) in input_offsets
            # println("putting $var[$(slice_inputs[var])] into inputs_packed[$s:$e]")
            inputs_packed[s:e] = inputs[var][slice_inputs[var]...]
        end

        for (var, (s, e)) in output_offsets
            # println("putting $var[$(slice_outputs[var])] into output_packed[$s:$e]")
            outputs_packed[s:e] = outputs[var][slice_outputs[var]...]
        end

        compute_diffable!(outputs_packed, inputs_packed)

        J = ForwardDiff.jacobian(compute_diffable!, outputs_packed, inputs_packed)

        for ((of_var, wrt_var), data) in partials
            @show of_var, wrt_var
            of_s, of_e = output_offsets[of_var][1], output_offsets[of_var][2]
            wrt_s, wrt_e = input_offsets[wrt_var][1], input_offsets[wrt_var][2]
            # The derivatives we want are in J[of_s:of_e, wrt_s:wrt_e]. What
            # derivatives are those? Well, I know they correspond to the
            # derivative of outputs[var][slice_outputs[var]...] with respect to
            # inputs[var][slice_inputs[var]...]. But the ordering of the inputs
            # and outputs doesn't necessarily line up with the way I put them in
            # the Jacobian.
            deriv = J[of_s:of_e, wrt_s:wrt_e]
            @show size(deriv)

            # Now, how do I get the derivatives in the right spot? Right now the
            # Jacobian is 2D. First step is to get the relevant output and input
            # subscripts. The output subscripts are the ones that are both in
            # ss_outputs[of_var] and in ss_outputs_unique.
            @show ss_outputs_unique
            out_ss = intersect(ss_outputs_unique, ss_outputs[of_var])
            @show out_ss

            # The input subscripts are the ones in ss_inputs[of_var] and
            # ss_inputs_unique.
            in_ss = intersect(ss_inputs_unique, ss_inputs[wrt_var])
            @show in_ss

            # Now we can get the derivative subscripts and size. Need to be
            # careful: I have to order the input and output subscripts the same
            # as how they were used to pack the data. 
            deriv_ss = vcat(out_ss, in_ss)
            # deriv_size = ones(Int64, n_ss - length(ss_outputs_common))
            # j = 1
            # for ss in ss_outputs_unique
            #     if ss in out_ss
            #         deriv_size[j] = ss_sizes[ss]
            #     end
            #     j += 1
            # end
            # for ss in ss_inputs_unique
            #     if ss in in_ss
            #         deriv_size[j] = ss_sizes[ss]
            #     end
            #     j += 1
            # end
            deriv_size = Int64[]
            for ss in ss_outputs[of_var]
                if ss in ss_outputs_unique
                    push!(deriv_size, ss_sizes[ss])
                end
            end
            for ss in ss_inputs[wrt_var]
                if ss in ss_inputs_unique
                    push!(deriv_size, ss_sizes[ss])
                end
            end
            @show deriv_size

            # Reshape the derivative. So the subscript order is vcat(out_ss,
            # in_ss).
            reshaped_deriv = reshape(deriv, deriv_size...)
            @show reshaped_deriv

            # So the next part is to place the reshaped derivative into the
            # partials that in the order of the original subscripts. What will
            # that be? Well, maybe I'll just have a consistant order? OK, let's
            # do that.
            # partials_slice = vcat(deriv_size, I)
            # partials[of_var, wrt_var][

            # Loop over things in "Jacobian order." So that means the unique
            # inputs and outputs. And I need to identify which unique input
            # subscripts correspond to this wrt, and which unique output
            # subscripts correspond to this of. Oh, already have that. So, hmm..
            # just need to loop over deriv_size.
            @show deriv_ss
            for deriv_idx in CartesianIndices(reshaped_deriv)
                @show deriv_idx
                deriv_ss_val = Dict{Symbol, Int64}()
                for (j, k) in enumerate(deriv_ss)
                    deriv_ss_val[k] = deriv_idx[j]
                end

                partials_idx = Int64[]
                for ss in ss_outputs_unique
                    if ss in ss_outputs[of_var]
                        push!(partials_idx, deriv_ss_val[ss])
                    else
                        push!(partials_idx, 1)
                    end
                end
                for ss in ss_inputs_unique
                    if ss in ss_inputs[wrt_var]
                        push!(partials_idx, deriv_ss_val[ss])
                    else
                        push!(partials_idx, 1)
                    end
                end
                for ss in ss_outputs_common
                    push!(partials_idx, i[ss])
                end
                @show partials_idx
                partials[of_var, wrt_var][partials_idx...] = reshaped_deriv[deriv_idx]
            end

        end

    end

end

main()
