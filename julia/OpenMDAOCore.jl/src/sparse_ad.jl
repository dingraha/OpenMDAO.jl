"""
    SparseADExplicitComp{InPlace,TAD,TCompute,TX,TY,TJ,TPrep,TXCS,TYCS,TAMD} <: AbstractExplicitComp{InPlace}

An `<:AbstractADExplicitComp` for sparse Jacobians.

# Fields
* `ad_backend::TAD`: `<:ADTypes.AutoSparse` automatic differentation "backend" library
* `compute_adable::TCompute`: function of the form `compute_adable(Y, X)` compatible with DifferentiationInterface.jl that performs the desired computation, where `Y` and `X` are `ComponentVector`s of outputs and inputs, respectively
* `X_ca::ComponentVector`: `ComponentVector` of inputs
* `Y_ca::ComponentVector`: `ComponentVector` of outputs
* `J_ca_sparse::ComponentMatrix`: Sparse `ComponentMatrix` of the Jacobian of `Y_ca` with respect to `X_ca`
* `units_dict::Dict{Symbol,String}`: mapping of variable names to units. Can be an empty `Dict` if units are not desired.
* `tags_dict::Dict{Symbol,Vector{String}`: mapping of variable names to `Vector`s of `String`s specifing variable tags.
* `shape_by_conn_dict::Dict{Symbol,Bool}`: mapping of variable names to `Bool` indicating if the variable shape should be determined dynamically by a connection.
* `prep::DifferentiationInterface.JacobianPrep`: `DifferentiationInterface.jl` "preparation" object
* `rcdict`: `Dict{Tuple{Symbol,Sympol}, Tuple{Vector{Int}, Vector{Int}}` mapping sub-Jacobians of the form `(:output_name, :input_name)` to `Vector`s of non-zero row and column indices (1-based)
* `X_ca::ComponentVector`: `ComplexF64` version of `X_ca` (for the complex-step method)
* `Y_ca::ComponentVector`: `ComplexF64` version of `Y_ca` (for the complex-step method)
"""
struct SparseADExplicitComp{InPlace,TAD,TCompute,TX,TY,TJ,TPrep,TXCS,TYCS} <: AbstractADExplicitComp{InPlace}
    ad_backend::TAD
    compute_adable::TCompute
    X_ca::TX
    Y_ca::TY
    J_ca_sparse::TJ
    prep::TPrep
    rcdict::Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}
    units_dict::Dict{Symbol,String}
    tags_dict::Dict{Symbol,Vector{String}}
    shape_by_conn_dict::Dict{Symbol,Bool}
    copy_shape_dict::Dict{Symbol,Symbol}
    X_ca_cs::TXCS
    Y_ca_cs::TYCS

    function SparseADExplicitComp{InPlace}(ad_backend, compute_adable, X_ca, Y_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs) where {InPlace}
        return new{
                                    InPlace, typeof(ad_backend), typeof(compute_adable), typeof(X_ca), typeof(Y_ca),
                                    typeof(J_ca_sparse), typeof(prep),
                                    typeof(X_ca_cs), typeof(Y_ca_cs)}(ad_backend,
                                                              compute_adable, X_ca,
                                                              Y_ca, J_ca_sparse,
                                                              prep, rcdict,
                                                              units_dict,
                                                              tags_dict,
                                                              shape_by_conn_dict,
                                                              copy_shape_dict,
                                                              X_ca_cs, Y_ca_cs)
    end
end

function SparseADExplicitComp{false}(ad_backend, compute_adable, X_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs)
    Y_ca = nothing
    Y_ca_cs = nothing
    return SparseADExplicitComp{false}(ad_backend, compute_adable, X_ca, Y_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs)
end

"""
    SparseADExplicitComp(ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create a `SparseADExplicitComp` from a user-defined function and output and input `ComponentVector`s.

# Positional Arguments
* `ad_backend`: `<:ADTypes.AutoSparse` automatic differentation "backend" library
* `f!`: function of the form `f!(Y_ca, X_ca, params)` which writes outputs to `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s)
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol
* `force_skip_prep`: if true, defer creating internal arrays and other structs until the user calls `update_prep!`
"""
function SparseADExplicitComp(ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AutoSparse}

    # Create a new user-defined function that captures the `params` argument.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_adable = let params=params
        (Y, X)->begin
            f!(Y, X, params)
            return nothing
        end
    end

    # Get the prep-related stuff.
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0) && (!force_skip_prep)
        prep, J_ca_sparse, rcdict, X_ca_cs, Y_ca_cs = _get_sparse_prep_stuff(ad_backend, compute_adable, Y_ca, X_ca)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        prep = J_ca_sparse = X_ca_cs = Y_ca_cs = nothing
        rcdict = Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}()
    end

    return SparseADExplicitComp{true}(ad_backend, compute_adable, X_ca, Y_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs)
end

"""
    SparseADExplicitComp(ad_backend, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create a `SparseADExplicitComp` from a user-defined function and output and input `ComponentVector`s.

# Positional Arguments
* `ad_backend`: `<:ADTypes.AutoSparse` automatic differentation "backend" library
* `f`: function of the form `Y_ca = f(X_ca, params)` which returns outputs `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s)
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol
* `force_skip_prep`: if true, defer creating internal arrays and other structs until the user calls `update_prep!`
"""
function SparseADExplicitComp(ad_backend::TAD, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AutoSparse}

    # Create a new user-defined function that captures the `params` argument.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_adable = let params=params
        (X,)->begin
            return f(X, params)
        end
    end

    Y_ca = compute_adable(X_ca)

    # Get the prep-related stuff.
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0) && (!force_skip_prep)
        prep, J_ca_sparse, rcdict, X_ca_cs = _get_sparse_prep_stuff(ad_backend, compute_adable, X_ca)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        # prep = DifferentiationInterface.NoJacobianPrep(DifferentiationInterface.signature(compute_adable, ad_backend, X_ca_full; strict=Val{true}()))
        # J_ca_sparse = ComponentMatrix{eltype(X_ca_full)}()
        # rcdict = Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}()
        # X_ca_cs = ComponentVector{ComplexF64}()
        prep = J_ca_sparse = X_ca_cs = nothing
        rcdict = Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}()
    end

    return SparseADExplicitComp{false}(ad_backend, compute_adable, X_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs)
end

get_rows_cols_dict(comp::SparseADExplicitComp) = comp.rcdict
get_jacobian_ca(comp::SparseADExplicitComp) = comp.J_ca_sparse

function _get_sparse_prep_stuff(ad_backend, f!, Y_ca, X_ca)
    # Need to "prepare" the backend.
    prep = DifferentiationInterface.prepare_jacobian(f!, Y_ca, ad_backend, X_ca)

    # Now I think I can get the sparse Jacobian from that.
    J_sparse = Float64.(SparseMatrixColorings.sparsity_pattern(prep))

    # Then use that sparse Jacobian to create the component matrix version.
    J_ca_sparse = ComponentMatrix(J_sparse, (only(getaxes(Y_ca,)), only(getaxes(X_ca))))

    # Get a dictionary describing the non-zero rows and cols for each subjacobian.
    rcdict = get_rows_cols_dict_from_sparsity(J_ca_sparse)

    # Create complex-valued versions of the X_ca_full and Y_ca_full arrays.
    X_ca_cs = similar(X_ca, ComplexF64)
    Y_ca_cs = similar(Y_ca, ComplexF64)

    return prep, J_ca_sparse, rcdict, X_ca_cs, Y_ca_cs
end

function _get_sparse_prep_stuff(ad_backend, f, X_ca)
    # Need to "prepare" the backend.
    prep = DifferentiationInterface.prepare_jacobian(f, ad_backend, X_ca)

    # Now I think I can get the sparse Jacobian from that.
    J_sparse = Float64.(SparseMatrixColorings.sparsity_pattern(prep))

    # Need the output component vector to define the axes of the Jacobian.
    Y_ca = f(X_ca)

    # Then use that sparse Jacobian to create the component matrix version.
    J_ca_sparse = ComponentMatrix(J_sparse, (only(getaxes(Y_ca,)), only(getaxes(X_ca))))

    # Get a dictionary describing the non-zero rows and cols for each subjacobian.
    rcdict = get_rows_cols_dict_from_sparsity(J_ca_sparse)

    # Create complex-valued versions of the X_ca_full and Y_ca_full arrays.
    X_ca_cs = similar(X_ca, ComplexF64)

    return prep, J_ca_sparse, rcdict, X_ca_cs
end

# function update_prep!(self::SparseADExplicitComp{true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
#
#     if (length(input_sizes) > 0) || (length(output_sizes) > 0)
#         X_ca_old = get_input_ca(self)
#         # For an out-of-place component, this will call the callback function on self.X_ca, which I think should be fine.
#         Y_ca_old = get_output_ca(self)
#
#         # Create a new versions of `X_ca_old` that have the correct sizes and default values.
#         X_ca = _resize_component_vector(X_ca_old, input_sizes)
#         Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
#
#         # Get the new sparsity stuff.
#         prep, J_ca_sparse, rcdict, X_ca_cs, Y_ca_cs = _get_sparse_prep_stuff(get_backend(self), get_callback(self), Y_ca, X_ca)
#
#         # Save everything in this struct.
#         self.X_ca = X_ca
#         self.Y_ca = Y_ca
#         self.J_ca_sparse = J_ca_sparse
#         self.prep = prep
#         self.rcdict = rcdict
#         self.X_ca_cs = X_ca_cs
#         self.Y_ca_cs = Y_ca_cs
#     end
#
#     return nothing
# end

function update_prep(self::SparseADExplicitComp{true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})

    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        # For an out-of-place component, this will call the callback function on self.X_ca, which I think should be fine.
        # Ah, but this is an in-place component anyway.
        Y_ca_old = get_output_ca(self)

        # Create a new versions of `X_ca_old` that have the correct sizes and default values.
        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        # Get the new sparsity stuff.
        ad_backend = get_backend(self)
        f! = get_callback(self)
        prep, J_ca_sparse, rcdict, X_ca_cs, Y_ca_cs = _get_sparse_prep_stuff(ad_backend, f!, Y_ca, X_ca)

        # Now just copy things over.
        units_dict = self.units_dict
        tags_dict = self.tags_dict
        shape_by_conn_dict = self.shape_by_conn_dict
        copy_shape_dict = self.copy_shape_dict

        self = SparseADExplicitComp{true}(ad_backend, f!, X_ca, Y_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs)
    end

    return self
end

# function update_prep!(self::SparseADExplicitComp{false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
#
#     if length(input_sizes) > 0
#         X_ca_old = get_input_ca(self)
#         # For an out-of-place component, this will call the callback function on self.X_ca, which I think should be fine.
#         # Y_ca_old = get_output_ca(self)
#
#         # Create a new versions of `X_ca_old` that have the correct sizes and default values.
#         X_ca = _resize_component_vector(X_ca_old, input_sizes)
#         # Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
#
#         # Get the new sparsity stuff.
#         prep, J_ca_sparse, rcdict, X_ca_cs = _get_sparse_prep_stuff(get_backend(self), get_callback(self), X_ca)
#
#         # Save everything in this struct.
#         self.X_ca = X_ca
#         self.J_ca_sparse = J_ca_sparse
#         self.prep = prep
#         self.rcdict = rcdict
#         self.X_ca_cs = X_ca_cs
#     end
#
#     return nothing
# end

function update_prep(self::SparseADExplicitComp{false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})

    if length(input_sizes) > 0
        X_ca_old = get_input_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        # Get the new sparsity stuff.
        ad_backend = get_backend(self)
        f = get_callback(self)
        prep, J_ca_sparse, rcdict, X_ca_cs = _get_sparse_prep_stuff(ad_backend, f, X_ca)

        # Now just copy things over.
        units_dict = self.units_dict
        tags_dict = self.tags_dict
        shape_by_conn_dict = self.shape_by_conn_dict
        copy_shape_dict = self.copy_shape_dict

        self = SparseADExplicitComp{false}(ad_backend, f, X_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs)
    end

    return self
end

function  _get_py_indices_non_flat(shape)
    # First, get the flattened 0-based indices.
    idx_flat = 0:(prod(shape)-1)

    # Now reshape it into the reversed dimenions, then permute the dimensions.
    # This will give us an array that has the shape indicated by the `shape` argument to this function, but filled with the appropriated indices for a zero-based, Python-ordered (aka row-major ordered) array.
    return PermutedDimsArray(reshape(idx_flat, reverse(shape)), length(shape):-1:1)
end

function  _get_py_indices(shape)
    idx_non_flat = _get_py_indices_non_flat(shape)
    # Now create a flattened view:
    return view(idx_non_flat, :)
end

function get_partials_data(self::SparseADExplicitComp)
    rcdict = get_rows_cols_dict(self)
    partials_data = Vector{OpenMDAOCore.PartialsData}()
    X_ca = get_input_ca(self)
    Y_ca = get_output_ca(self)
    for (output_name, input_name) in keys(rcdict)
        rows, cols = rcdict[output_name, input_name]

        # Create an array that has the same shape as the input or output but with Python flat indices™ as values.
        input_idx_py = _get_py_indices(size(X_ca[input_name]))
        output_idx_py = _get_py_indices(size(Y_ca[output_name]))

        # Translate the Julia-ordered, 1-based rows and cols to Python-ordered, 0-based rows and cols.
        cols0based = getindex.(Ref(input_idx_py), cols)
        rows0based = getindex.(Ref(output_idx_py), rows)

        push!(partials_data, OpenMDAOCore.PartialsData(string(output_name), string(input_name); rows=rows0based, cols=cols0based))
    end

    return partials_data
end

function setup_partials(self::SparseADExplicitComp, input_sizes, output_sizes)

    input_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in input_sizes)
    output_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in output_sizes)

    self_new = update_prep(self, input_sizes_ca, output_sizes_ca)

    # Now finally get the partials data.
    return self_new, get_partials_data(self_new)
end

_maybe_nonzeros(A::AbstractArray) = A
_maybe_nonzeros(A::AbstractSparseArray) = nonzeros(A)
_maybe_nonzeros(A::Base.ReshapedArray{T,N,P}) where {T,N,P<:AbstractSparseArray} = nonzeros(parent(A))

function OpenMDAOCore.compute_partials!(self::SparseADExplicitComp{true}, inputs, partials)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    # Get the Jacobian.
    f! = get_callback(self)
    Y_ca = get_output_ca(self)
    J_ca_sparse = get_jacobian_ca(self)
    prep = get_prep(self)
    ad_backend = get_backend(self)
    DifferentiationInterface.jacobian!(f!, Y_ca, J_ca_sparse, prep, ad_backend, X_ca)

    # Extract the derivatives from `J_ca_sparse` and put them in `partials`.
    raxis, caxis = getaxes(J_ca_sparse)
    rcdict = get_rows_cols_dict(self)
    for oname in keys(raxis)
        for iname in keys(caxis)
            # Grab the subjacobian we're interested in.
            Jsub_in = @view(J_ca_sparse[oname, iname])

            # Need to reshape the subjacobian to correspond to the rows and cols.
            nrows = length(raxis[oname])
            ncols = length(caxis[iname])
            Jsub_in_reshape = reshape(Jsub_in, nrows, ncols)

            # Grab the entry in partials we're interested in, and write the data we want to it.
            rows, cols = rcdict[oname, iname]

            # OpenMDAO might not ask for all the partials, and so all combination of output/input keys might not be present in `partials`.
            local Jsub_out
            try
                Jsub_out = partials[string(oname), string(iname)]
            catch e
                if !isa(e, KeyError)
                    rethrow()
                end
            else
                # This will get a vector of the non-zero entries of the sparse sub-Jacobian if it's actually sparse, or just a reference to the flattened vector of the dense sub-Jacobian otherwise.
                Jsub_out_vec = _maybe_nonzeros(Jsub_out)

                # Now write the non-zero entries to Jsub_out_vec.
                Jsub_out_vec .= getindex.(Ref(Jsub_in_reshape), rows, cols)
            end

        end
    end

    return nothing
end

function OpenMDAOCore.compute_partials!(self::SparseADExplicitComp{false}, inputs, partials)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    # Get the Jacobian.
    f = get_callback(self)
    # Y_ca = get_output_ca(self)
    J_ca_sparse = get_jacobian_ca(self)
    prep = get_prep(self)
    ad_backend = get_backend(self)
    DifferentiationInterface.jacobian!(f, J_ca_sparse, prep, ad_backend, X_ca)

    # Extract the derivatives from `J_ca_sparse` and put them in `partials`.
    raxis, caxis = getaxes(J_ca_sparse)
    rcdict = get_rows_cols_dict(self)
    for oname in keys(raxis)
        for iname in keys(caxis)
            # Grab the subjacobian we're interested in.
            Jsub_in = @view(J_ca_sparse[oname, iname])
            
            # Need to reshape the subjacobian to correspond to the rows and cols.
            nrows = length(raxis[oname])
            ncols = length(caxis[iname])
            Jsub_in_reshape = reshape(Jsub_in, nrows, ncols)

            # Grab the entry in partials we're interested in, and write the data we want to it.
            rows, cols = rcdict[oname, iname]

            # OpenMDAO might not ask for all the partials, and so all combination of output/input keys might not be present in `partials`.
            local Jsub_out
            try
                Jsub_out = partials[string(oname), string(iname)]
            catch e
                if !isa(e, KeyError)
                    rethrow()
                end
            else
                # This will get a vector of the non-zero entries of the sparse sub-Jacobian if it's actually sparse, or just a reference to the flattened vector of the dense sub-Jacobian otherwise.
                Jsub_out_vec = _maybe_nonzeros(Jsub_out)

                # Now write the non-zero entries to Jsub_out_vec.
                Jsub_out_vec .= getindex.(Ref(Jsub_in_reshape), rows, cols)
            end
        end
    end

    return nothing
end

has_setup_partials(self::SparseADExplicitComp) = true
has_compute_partials(self::SparseADExplicitComp) = true
has_compute_jacvec_product(self::SparseADExplicitComp) = false
