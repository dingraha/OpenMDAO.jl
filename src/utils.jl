export get_rows_cols, struct2array!, structs2array!, array2struct!, array2structs!, nfieldsoftype

function get_rows_cols(ss_sizes, of_ss, wrt_ss)
    # Get the output subscript, which will start with the of_ss, then the
    # wrt_ss with the subscripts common to both removed.
    # deriv_ss = of_ss + "".join(set(wrt_ss) - set(of_ss))
    deriv_ss = vcat(of_ss, setdiff(wrt_ss, of_ss))

    # Reverse the subscripts so they work with column-major ordering.
    of_ss = reverse(of_ss)
    wrt_ss = reverse(wrt_ss)
    deriv_ss = reverse(deriv_ss)

    # Get the shape of the output variable (the "of"), the input variable
    # (the "wrt"), and the derivative (the Jacobian).
    of_shape = Tuple(ss_sizes[s] for s in of_ss)
    wrt_shape = Tuple(ss_sizes[s] for s in wrt_ss)
    deriv_shape = Tuple(ss_sizes[s] for s in deriv_ss)

    # Invert deriv_ss: get a dictionary that goes from subscript to index
    # dimension.
    deriv_ss2idx = Dict(ss=>i for (i, ss) in enumerate(deriv_ss))

    # This is the equivalent of the Python code
    #   a = np.arange(np.prod(of_shape)).reshape(of_shape)
    #   b = np.arange(np.prod(wrt_shape)).reshape(wrt_shape)
    # but in column major order, which is OK, since we've reversed the order of
    # of_shape and wrt_shape above.
    a = reshape(0:prod(of_shape)-1, of_shape)
    b = reshape(0:prod(wrt_shape)-1, wrt_shape)

    rows = Array{Int}(undef, deriv_shape)
    cols = Array{Int}(undef, deriv_shape)
    for deriv_idx in CartesianIndices(deriv_shape)
        # Go from the jacobian index to the of and wrt indices.
        of_idx = [deriv_idx[deriv_ss2idx[ss]] for ss in of_ss]
        wrt_idx = [deriv_idx[deriv_ss2idx[ss]] for ss in wrt_ss]

        # Get the flattened index for the output and input.
        rows[deriv_idx] = a[of_idx...]
        cols[deriv_idx] = b[wrt_idx...]
    end

    # Return flattened versions of the rows and cols arrays.
    return rows[:], cols[:]
end

get_rows_cols(; ss_sizes, of_ss, wrt_ss) = get_rows_cols(ss_sizes, of_ss, wrt_ss)

# https://discourse.julialang.org/t/how-to-write-a-fast-loop-through-structure-fields/22535
@generated function struct2array!(a::AbstractArray{T}, dt::DT, i::Integer=1) where {T, DT}
    assignments = Expr[]
    for (name, S) in zip(fieldnames(DT), DT.types)
        if S == T
            push!(assignments, :(a[i] = dt.$name))
            push!(assignments, :(i += 1))
        end
    end
    quote $(assignments...) end
end

function structs2array!(a::AbstractArray{T}, structs, i::Integer=1) where {T}
    for dt in structs
      i = struct2array!(a, dt, i)
    end
    return i
end

# https://discourse.julialang.org/t/how-to-write-a-fast-loop-through-structure-fields/22535
@generated function array2struct!(dt::DT, a::AbstractArray{T}, i::Integer=1) where {T, DT}
    assignments = Expr[]
    for (name, S) in zip(fieldnames(DT), DT.types)
        if S == T
            push!(assignments, :(dt.$name = a[i]))
            push!(assignments, :(i += 1))
        end
    end
    quote $(assignments...) end
end

function array2structs!(structs, a::AbstractArray{T}, i::Integer=1) where {T}
    for dt in structs
      i = array2struct!(dt, a, i)
    end
    return i
end

function nfieldsoftype(T::DataType, dt::DT) where {DT}
    return count(S -> S==T, DT.types)
end

function nfieldsoftype(T::DataType, structs::AbstractArray)
    i = 0
    for dt in structs
        i += nfieldsoftype(T, dt)
    end
    return i
end
