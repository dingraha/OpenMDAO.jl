# Convenience constructors for `ADExplicitComp`, inspired by
# OpenMDAO4Core.jl's `create_explicit_component`.
#
# These wrap the flavor-specific `ADExplicitComp(flavor, ...)` constructors with
# an additional validation step: for matrix-free flavors, the AD backend must
# support the required derivative mode (JVP for `MatrixFreeForwardFlavor`, VJP
# for `MatrixFreeReverseFlavor`); otherwise an `ArgumentError` is thrown with a
# helpful message. Dense and Sparse flavors have no extra check (a non-sparse
# backend passed with `SparseFlavor` yields a `MethodError` from the delegate's
# `TAD<:ADTypes.AutoSparse` constraint, matching direct `ADExplicitComp` use).

"""
    create_explicit_component(flavor::DerivativeFlavor, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create an in-place [`ADExplicitComp`](@ref) from a user-defined function and output and input `ComponentVector`s.

For the matrix-free flavors, `ad_backend` is validated against the flavor's required derivative mode:
* [`MatrixFreeForwardFlavor`](@ref) requires the backend to support JVPs (pushforwards).
* [`MatrixFreeReverseFlavor`](@ref) requires the backend to support VJPs (pullbacks).

# Positional Arguments
* `flavor`: [`DenseFlavor`](@ref), [`MatrixFreeForwardFlavor`](@ref), [`MatrixFreeReverseFlavor`](@ref), or [`SparseFlavor`](@ref)
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentiation "backend" library
* `f!`: function of the form `f!(Y_ca, X_ca, params)` which writes outputs to `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
"""
function create_explicit_component(flavor::DerivativeFlavor, ad_backend::TAD,
        f!::Function, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(),
        force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    if flavor isa MatrixFreeForwardFlavor
        can_jvp(ad_backend) || throw(ArgumentError("AD backend $(ad_backend) does not support JVPs (pushforwards), which are required for MatrixFreeForwardFlavor"))
    elseif flavor isa MatrixFreeReverseFlavor
        can_vjp(ad_backend) || throw(ArgumentError("AD backend $(ad_backend) does not support VJPs (pullbacks), which are required for MatrixFreeReverseFlavor"))
    end
    return ADExplicitComp(flavor, ad_backend, f!, Y_ca, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict,
        shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict,
        force_skip_prep=force_skip_prep)
end

"""
    create_explicit_component(flavor::DerivativeFlavor, ad_backend, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create an out-of-place [`ADExplicitComp`](@ref) from a user-defined function and input `ComponentVector`.

For the matrix-free flavors, `ad_backend` is validated against the flavor's required derivative mode:
* [`MatrixFreeForwardFlavor`](@ref) requires the backend to support JVPs (pushforwards).
* [`MatrixFreeReverseFlavor`](@ref) requires the backend to support VJPs (pullbacks).

# Positional Arguments
* `flavor`: [`DenseFlavor`](@ref), [`MatrixFreeForwardFlavor`](@ref), [`MatrixFreeReverseFlavor`](@ref), or [`SparseFlavor`](@ref)
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentiation "backend" library
* `f`: function of the form `Y_ca = f(X_ca, params)` which returns outputs `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the second argument to `f`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
"""
function create_explicit_component(flavor::DerivativeFlavor, ad_backend::TAD,
        f::Function, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(),
        force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    if flavor isa MatrixFreeForwardFlavor
        can_jvp(ad_backend) || throw(ArgumentError("AD backend $(ad_backend) does not support JVPs (pushforwards), which are required for MatrixFreeForwardFlavor"))
    elseif flavor isa MatrixFreeReverseFlavor
        can_vjp(ad_backend) || throw(ArgumentError("AD backend $(ad_backend) does not support VJPs (pullbacks), which are required for MatrixFreeReverseFlavor"))
    end
    return ADExplicitComp(flavor, ad_backend, f, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict,
        shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict,
        force_skip_prep=force_skip_prep)
end

# ---------------------------------------------------------------------------
# Convenience constructors for `ADImplicitComp`
# ---------------------------------------------------------------------------

"""
    create_implicit_component(flavor::DerivativeFlavor, ::Val{true}, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, force_skip_prep=false)

Create an in-place [`ADImplicitComp`](@ref) from a user-defined function and state/output and input `ComponentVector`s.

For the matrix-free flavors, `ad_backend` is validated against the flavor's required derivative mode:
* [`MatrixFreeForwardFlavor`](@ref) requires the backend to support JVPs (pushforwards).
* [`MatrixFreeReverseFlavor`](@ref) requires the backend to support VJPs (pullbacks).

# Positional Arguments
* `flavor`: [`DenseFlavor`](@ref), [`SparseFlavor`](@ref), [`MatrixFreeForwardFlavor`](@ref), or [`MatrixFreeReverseFlavor`](@ref)
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentiation "backend" library
* `f!`: function of the form `f!(R_ca, Y_ca, X_ca, params)` which writes residuals to `R_ca` using states/outputs `Y_ca`, inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of states/outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the fourth argument to `f!`. Could be anything, or `nothing`, but the derivatives of `R_ca` with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
"""
function create_implicit_component(flavor::DerivativeFlavor, ::Val{true}, ad_backend::TAD,
        f!::Function, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    if flavor isa MatrixFreeForwardFlavor
        can_jvp(ad_backend) || throw(ArgumentError("AD backend $(ad_backend) does not support JVPs (pushforwards), which are required for MatrixFreeForwardFlavor"))
    elseif flavor isa MatrixFreeReverseFlavor
        can_vjp(ad_backend) || throw(ArgumentError("AD backend $(ad_backend) does not support VJPs (pullbacks), which are required for MatrixFreeReverseFlavor"))
    end
    return ADImplicitComp(flavor, Val(true), ad_backend, f!, Y_ca, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict, shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict, force_skip_prep=force_skip_prep)
end

"""
    create_implicit_component(flavor::DerivativeFlavor, ::Val{false}, ad_backend, f, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, force_skip_prep=false)

Create an out-of-place [`ADImplicitComp`](@ref) from a user-defined function and state/output and input `ComponentVector`s.

For the matrix-free flavors, `ad_backend` is validated against the flavor's required derivative mode:
* [`MatrixFreeForwardFlavor`](@ref) requires the backend to support JVPs (pushforwards).
* [`MatrixFreeReverseFlavor`](@ref) requires the backend to support VJPs (pullbacks).

# Positional Arguments
* `flavor`: [`DenseFlavor`](@ref), [`SparseFlavor`](@ref), [`MatrixFreeForwardFlavor`](@ref), or [`MatrixFreeReverseFlavor`](@ref)
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentiation "backend" library
* `f`: function of the form `R_ca = f(Y_ca, X_ca, params)` which returns residuals `R_ca` using states/outputs `Y_ca`, inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of states/outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f`. Could be anything, or `nothing`, but the derivatives of `R_ca` with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
"""
function create_implicit_component(flavor::DerivativeFlavor, ::Val{false}, ad_backend::TAD,
        f::Function, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    if flavor isa MatrixFreeForwardFlavor
        can_jvp(ad_backend) || throw(ArgumentError("AD backend $(ad_backend) does not support JVPs (pushforwards), which are required for MatrixFreeForwardFlavor"))
    elseif flavor isa MatrixFreeReverseFlavor
        can_vjp(ad_backend) || throw(ArgumentError("AD backend $(ad_backend) does not support VJPs (pullbacks), which are required for MatrixFreeReverseFlavor"))
    end
    return ADImplicitComp(flavor, Val(false), ad_backend, f, Y_ca, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict, shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict, force_skip_prep=force_skip_prep)
end