# Matrix-free explicit AD component: constructors and methods, split into
# `MatrixFreeForwardFlavor` (pushforward / JVP) and `MatrixFreeReverseFlavor`
# (pullback / VJP).
#
# The `ADExplicitComp{<:MatrixFreeFlavor, ...}` *type* and the shared accessors
# (`get_dinput_ca`, `get_doutput_ca`, `has_*`) are declared in `abstract_ad.jl`.
# This file provides the flavor-specific constructors, `_update_prep`,
# `setup_partials`, and `compute_jacvec_product!` (+ internals).

using ADTypes: ADTypes
using ComponentArrays: ComponentVector
using DifferentiationInterface: DifferentiationInterface

# ---------------------------------------------------------------------------
# Flavor auto-detection (replaces the old `force_mode=""` behavior).
# ---------------------------------------------------------------------------

# Resolve the old `force_mode` argument to a concrete matrix-free flavor.
# `force_mode="fwd"` -> `MatrixFreeForwardFlavor`; `"rev"` -> `MatrixFreeReverseFlavor`;
# `""` -> pick based on the backend's pushforward performance (preferring forward),
# matching the previous runtime auto-detection.
function _resolve_matrix_free_flavor(ad_backend, force_mode::AbstractString="")
    if force_mode == "fwd"
        return MatrixFreeForwardFlavor()
    elseif force_mode == "rev"
        return MatrixFreeReverseFlavor()
    elseif force_mode == ""
        if DifferentiationInterface.pushforward_performance(ad_backend) isa DifferentiationInterface.PushforwardFast
            return MatrixFreeForwardFlavor()
        else
            return MatrixFreeReverseFlavor()
        end
    else
        throw(ArgumentError("force_mode argument should be one of `\"\"`, \"fwd\", or \"rev\" but is $(force_mode)"))
    end
end

# ---------------------------------------------------------------------------
# Prep builders. `force_skip_prep` replaces the old `disable_prep` flag.
# ---------------------------------------------------------------------------

function _get_matrix_free_forward_prep_in_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep::Bool)
    dX_ca = similar(X_ca)
    dY_ca = similar(Y_ca)
    if force_skip_prep
        prep = nothing
    else
        prep = DifferentiationInterface.prepare_pushforward(compute_adable, Y_ca, ad_backend, X_ca, (dX_ca,))
    end
    X_ca_cs = similar(X_ca, Complex{eltype(X_ca)})
    Y_ca_cs = similar(Y_ca, Complex{eltype(Y_ca)})
    return MatrixFreeDerivPrep(dX_ca, dY_ca, prep), X_ca_cs, Y_ca_cs
end

function _get_matrix_free_reverse_prep_in_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep::Bool)
    dX_ca = similar(X_ca)
    dY_ca = similar(Y_ca)
    if force_skip_prep
        prep = nothing
    else
        prep = DifferentiationInterface.prepare_pullback(compute_adable, Y_ca, ad_backend, X_ca, (dY_ca,))
    end
    X_ca_cs = similar(X_ca, Complex{eltype(X_ca)})
    Y_ca_cs = similar(Y_ca, Complex{eltype(Y_ca)})
    return MatrixFreeDerivPrep(dX_ca, dY_ca, prep), X_ca_cs, Y_ca_cs
end

function _get_matrix_free_forward_prep_out_of_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep::Bool)
    dX_ca = similar(X_ca)
    dY_ca = similar(Y_ca)
    if force_skip_prep
        prep = nothing
    else
        prep = DifferentiationInterface.prepare_pushforward(compute_adable, ad_backend, X_ca, (dX_ca,))
    end
    X_ca_cs = similar(X_ca, Complex{eltype(X_ca)})
    return MatrixFreeDerivPrep(dX_ca, dY_ca, prep), X_ca_cs
end

function _get_matrix_free_reverse_prep_out_of_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep::Bool)
    dX_ca = similar(X_ca)
    dY_ca = similar(Y_ca)
    if force_skip_prep
        prep = nothing
    else
        prep = DifferentiationInterface.prepare_pullback(compute_adable, ad_backend, X_ca, (dY_ca,))
    end
    X_ca_cs = similar(X_ca, Complex{eltype(X_ca)})
    return MatrixFreeDerivPrep(dX_ca, dY_ca, prep), X_ca_cs
end

# ---------------------------------------------------------------------------
# Constructors.
# ---------------------------------------------------------------------------

"""
    ADExplicitComp(::MatrixFreeForwardFlavor, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=..., tags_dict=..., shape_by_conn_dict=..., copy_shape_dict=..., force_skip_prep=false)

Create an in-place [`MatrixFreeForwardFlavor`](@ref) [`ADExplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pushforward!` (JVPs).
"""
function ADExplicitComp(::MatrixFreeForwardFlavor, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

    compute_adable = _make_compute_adable(Val(true), f!, params)

    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_matrix_free_forward_prep_in_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        # The prep is set to `nothing`; the prep-less DifferentiationInterface
        # methods are used at runtime until OpenMDAO's `setup_partials` call
        # creates the real prep.
        dX_ca = ComponentVector{eltype(X_ca)}()
        dY_ca = ComponentVector{eltype(Y_ca)}()
        X_ca_cs = ComponentVector{ComplexF64}()
        Y_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dX_ca, dY_ca, nothing)
    end

    return ADExplicitComp{MatrixFreeForwardFlavor, true}(ad_backend, f!, params, compute_adable, X_ca, Y_ca, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs)
end

"""
    ADExplicitComp(::MatrixFreeReverseFlavor, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=..., tags_dict=..., shape_by_conn_dict=..., copy_shape_dict=..., force_skip_prep=false)

Create an in-place [`MatrixFreeReverseFlavor`](@ref) [`ADExplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pullback!` (VJPs).
"""
function ADExplicitComp(::MatrixFreeReverseFlavor, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

    compute_adable = _make_compute_adable(Val(true), f!, params)

    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_matrix_free_reverse_prep_in_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        # The prep is set to `nothing`; the prep-less DifferentiationInterface
        # methods are used at runtime until OpenMDAO's `setup_partials` call
        # creates the real prep.
        dX_ca = ComponentVector{eltype(X_ca)}()
        dY_ca = ComponentVector{eltype(Y_ca)}()
        X_ca_cs = ComponentVector{ComplexF64}()
        Y_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dX_ca, dY_ca, nothing)
    end

    return ADExplicitComp{MatrixFreeReverseFlavor, true}(ad_backend, f!, params, compute_adable, X_ca, Y_ca, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs)
end

"""
    ADExplicitComp(::MatrixFreeForwardFlavor, ad_backend, f, X_ca::ComponentVector; params=nothing, units_dict=..., tags_dict=..., shape_by_conn_dict=..., copy_shape_dict=..., force_skip_prep=false)

Create an out-of-place [`MatrixFreeForwardFlavor`](@ref) [`ADExplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pushforward!` (JVPs).
"""
function ADExplicitComp(::MatrixFreeForwardFlavor, ad_backend, f, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

    compute_adable = _make_compute_adable(Val(false), f, params)

    Y_ca = compute_adable(X_ca)

    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, X_ca_cs = _get_matrix_free_forward_prep_out_of_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        # The prep is set to `nothing`; the prep-less DifferentiationInterface
        # methods are used at runtime until OpenMDAO's `setup_partials` call
        # creates the real prep.
        dX_ca = ComponentVector{eltype(X_ca)}()
        dY_ca = ComponentVector{eltype(Y_ca)}()
        X_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dX_ca, dY_ca, nothing)
    end

    Y_ca_cs = nothing
    return ADExplicitComp{MatrixFreeForwardFlavor, false}(ad_backend, f, params, compute_adable, X_ca, nothing, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs)
end

"""
    ADExplicitComp(::MatrixFreeReverseFlavor, ad_backend, f, X_ca::ComponentVector; params=nothing, units_dict=..., tags_dict=..., shape_by_conn_dict=..., copy_shape_dict=..., force_skip_prep=false)

Create an out-of-place [`MatrixFreeReverseFlavor`](@ref) [`ADExplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pullback!` (VJPs).
"""
function ADExplicitComp(::MatrixFreeReverseFlavor, ad_backend, f, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

    compute_adable = _make_compute_adable(Val(false), f, params)

    Y_ca = compute_adable(X_ca)

    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, X_ca_cs = _get_matrix_free_reverse_prep_out_of_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        # The prep is set to `nothing`; the prep-less DifferentiationInterface
        # methods are used at runtime until OpenMDAO's `setup_partials` call
        # creates the real prep.
        dX_ca = ComponentVector{eltype(X_ca)}()
        dY_ca = ComponentVector{eltype(Y_ca)}()
        X_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dX_ca, dY_ca, nothing)
    end

    Y_ca_cs = nothing
    return ADExplicitComp{MatrixFreeReverseFlavor, false}(ad_backend, f, params, compute_adable, X_ca, nothing, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs)
end

# ---------------------------------------------------------------------------
# _update_prep / setup_partials / get_partials_data
# ---------------------------------------------------------------------------

function _update_prep(self::ADExplicitComp{MatrixFreeForwardFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        Y_ca_old = get_output_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        ad_backend = get_backend(self)
        f! = _make_compute_adable(Val(true), self.func, self.params)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_matrix_free_forward_prep_in_place(ad_backend, f!, Y_ca, X_ca, false)

        self = ADExplicitComp{MatrixFreeForwardFlavor, true}(ad_backend, self.func, self.params, f!, X_ca, Y_ca, deriv_prep,
            self.units_dict, self.tags_dict, self.shape_by_conn_dict, self.copy_shape_dict, X_ca_cs, Y_ca_cs)
    end
    return self
end

function _update_prep(self::ADExplicitComp{MatrixFreeReverseFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        Y_ca_old = get_output_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        ad_backend = get_backend(self)
        f! = _make_compute_adable(Val(true), self.func, self.params)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_matrix_free_reverse_prep_in_place(ad_backend, f!, Y_ca, X_ca, false)

        self = ADExplicitComp{MatrixFreeReverseFlavor, true}(ad_backend, self.func, self.params, f!, X_ca, Y_ca, deriv_prep,
            self.units_dict, self.tags_dict, self.shape_by_conn_dict, self.copy_shape_dict, X_ca_cs, Y_ca_cs)
    end
    return self
end

function _update_prep(self::ADExplicitComp{MatrixFreeForwardFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        Y_ca_old = get_output_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        ad_backend = get_backend(self)
        f = _make_compute_adable(Val(false), self.func, self.params)
        deriv_prep, X_ca_cs = _get_matrix_free_forward_prep_out_of_place(ad_backend, f, Y_ca, X_ca, false)

        self = ADExplicitComp{MatrixFreeForwardFlavor, false}(ad_backend, self.func, self.params, f, X_ca, nothing, deriv_prep,
            self.units_dict, self.tags_dict, self.shape_by_conn_dict, self.copy_shape_dict, X_ca_cs, nothing)
    end
    return self
end

function _update_prep(self::ADExplicitComp{MatrixFreeReverseFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        Y_ca_old = get_output_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        ad_backend = get_backend(self)
        f = _make_compute_adable(Val(false), self.func, self.params)
        deriv_prep, X_ca_cs = _get_matrix_free_reverse_prep_out_of_place(ad_backend, f, Y_ca, X_ca, false)

        self = ADExplicitComp{MatrixFreeReverseFlavor, false}(ad_backend, self.func, self.params, f, X_ca, nothing, deriv_prep,
            self.units_dict, self.tags_dict, self.shape_by_conn_dict, self.copy_shape_dict, X_ca_cs, nothing)
    end
    return self
end

# Matrix-free components don't declare any partials (the Jacobian is never assembled).
function get_partials_data(self::ADExplicitComp{<:MatrixFreeFlavor})
    return Vector{PartialsData}()
end

function setup_partials(self::ADExplicitComp{<:MatrixFreeFlavor}, input_sizes, output_sizes)
    input_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in input_sizes)
    output_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in output_sizes)

    self_new = _update_prep(self, input_sizes_ca, output_sizes_ca)

    return self_new, get_partials_data(self_new)
end

# ---------------------------------------------------------------------------
# compute_jacvec_product! + internals (pushforward / pullback).
# ---------------------------------------------------------------------------

function _compute_pushforward!(self::ADExplicitComp{MatrixFreeForwardFlavor, true}, inputs, d_inputs, d_outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    # d_inputs has the derivative of each input wrt some upstream input.
    dX_ca = get_dinput_ca(self)
    for iname in keys(dX_ca)
        iname_str = string(iname)
        if iname_str in keys(d_inputs)
            @view(dX_ca[iname]) .= d_inputs[iname_str]
        else
            @view(dX_ca[iname]) .= zero(eltype(dX_ca))
        end
    end

    # The AD library will need the output component array to do the Jacobian-vector product.
    Y_ca = get_output_ca(self)

    # We'll write the result of the Jacobian-vector product to a different component array.
    dY_ca = get_doutput_ca(self)

    # This is the function that will actually do the computation.
    compute_adable = get_callback(self)

    # We stored the "preparation" for reusing.
    prep = get_prep(self)

    # Get the AD backend.
    backend = get_backend(self)

    # Now actually do the Jacobian-vector product.
    if prep === nothing
        isempty(dX_ca) && error("The DifferentiationInterface prep for this component has not been created. This happens when the component is created with `force_skip_prep=true` or with `shape_by_conn`/`copy_shape` variables before OpenMDAO has finished setting up the problem (the latter case is resolved automatically during problem setup).")
        # No prep available: fall back to the prep-less DifferentiationInterface
        # methods, which prepare internally on each call.
    DifferentiationInterface.pushforward!(compute_adable, Y_ca, (dY_ca,), backend, X_ca, (dX_ca,))
    else
    DifferentiationInterface.pushforward!(compute_adable, Y_ca, (dY_ca,), prep, backend, X_ca, (dX_ca,))
    end

    # Now copy the output derivatives to `d_outputs`:
    for oname in keys(dY_ca)
        oname_str = string(oname)
        if typeof(d_outputs[oname_str]) <: AbstractArray
            d_outputs[oname_str] .+= @view(dY_ca[oname])
        else
            d_outputs[oname_str] += only(dY_ca[oname])
        end
    end

    return nothing
end

function _compute_pushforward!(self::ADExplicitComp{MatrixFreeForwardFlavor, false}, inputs, d_inputs, d_outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    dX_ca = get_dinput_ca(self)
    for iname in keys(dX_ca)
        iname_str = string(iname)
        if iname_str in keys(d_inputs)
            @view(dX_ca[iname]) .= d_inputs[iname_str]
        else
            @view(dX_ca[iname]) .= zero(eltype(dX_ca))
        end
    end

    # We'll write the result of the Jacobian-vector product to a different component array.
    dY_ca = get_doutput_ca(self)

    # This is the function that will actually do the computation.
    compute_adable = get_callback(self)

    # We stored the "preparation" for reusing.
    prep = get_prep(self)

    # Get the AD backend.
    backend = get_backend(self)

    # Now actually do the Jacobian-vector product.
    if prep === nothing
        isempty(dX_ca) && error("The DifferentiationInterface prep for this component has not been created. This happens when the component is created with `force_skip_prep=true` or with `shape_by_conn`/`copy_shape` variables before OpenMDAO has finished setting up the problem (the latter case is resolved automatically during problem setup).")
        # No prep available: fall back to the prep-less DifferentiationInterface
        # methods, which prepare internally on each call.
    DifferentiationInterface.pushforward!(compute_adable, (dY_ca,), backend, X_ca, (dX_ca,))
    else
    DifferentiationInterface.pushforward!(compute_adable, (dY_ca,), prep, backend, X_ca, (dX_ca,))
    end

    # Now copy the output derivatives to `d_outputs`:
    for oname in keys(dY_ca)
        oname_str = string(oname)
        if typeof(d_outputs[oname_str]) <: AbstractArray
            d_outputs[oname_str] .+= @view(dY_ca[oname])
        else
            d_outputs[oname_str] += only(dY_ca[oname])
        end
    end

    return nothing
end

function _compute_pullback!(self::ADExplicitComp{MatrixFreeReverseFlavor, true}, inputs, d_inputs, d_outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    dY_ca = get_doutput_ca(self)
    for oname in keys(dY_ca)
        oname_str = string(oname)
        if oname_str in keys(d_outputs)
            @view(dY_ca[oname]) .= d_outputs[oname_str]
        else
            @view(dY_ca[oname]) .= zero(eltype(dY_ca))
        end
    end

    # The AD library will need the output component array to do the Jacobian-vector product.
    Y_ca = get_output_ca(self)

    # We'll write the result of the vector-Jacobian product to a different component array.
    dX_ca = get_dinput_ca(self)

    # This is the function that will actually do the computation.
    compute_adable = get_callback(self)

    # Need the AD backend.
    backend = get_backend(self)

    # We stored the "preparation" for reusing.
    prep = get_prep(self)

    # Now actually do the Jacobian-vector product.
    if prep === nothing
        isempty(dX_ca) && error("The DifferentiationInterface prep for this component has not been created. This happens when the component is created with `force_skip_prep=true` or with `shape_by_conn`/`copy_shape` variables before OpenMDAO has finished setting up the problem (the latter case is resolved automatically during problem setup).")
        # No prep available: fall back to the prep-less DifferentiationInterface
        # methods, which prepare internally on each call.
    DifferentiationInterface.pullback!(compute_adable, Y_ca, (dX_ca,), backend, X_ca, (dY_ca,))
    else
    DifferentiationInterface.pullback!(compute_adable, Y_ca, (dX_ca,), prep, backend, X_ca, (dY_ca,))
    end

    # Now copy the input derivatives to `d_inputs`:
    for iname in keys(dX_ca)
        iname_str = string(iname)
        if typeof(d_inputs[iname_str]) <: AbstractArray
            d_inputs[iname_str] .+= @view(dX_ca[iname])
        else
            d_inputs[iname_str] += only(dX_ca[iname])
        end
    end

    return nothing
end

function _compute_pullback!(self::ADExplicitComp{MatrixFreeReverseFlavor, false}, inputs, d_inputs, d_outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    dY_ca = get_doutput_ca(self)
    for oname in keys(dY_ca)
        oname_str = string(oname)
        if oname_str in keys(d_outputs)
            @view(dY_ca[oname]) .= d_outputs[oname_str]
        else
            @view(dY_ca[oname]) .= zero(eltype(dY_ca))
        end
    end

    # We'll write the result of the vector-Jacobian product to a different component array.
    dX_ca = get_dinput_ca(self)

    # This is the function that will actually do the computation.
    compute_adable = get_callback(self)

    # Need the AD backend.
    backend = get_backend(self)

    # We stored the "preparation" for reusing.
    prep = get_prep(self)

    # Now actually do the Jacobian-vector product.
    if prep === nothing
        isempty(dX_ca) && error("The DifferentiationInterface prep for this component has not been created. This happens when the component is created with `force_skip_prep=true` or with `shape_by_conn`/`copy_shape` variables before OpenMDAO has finished setting up the problem (the latter case is resolved automatically during problem setup).")
        # No prep available: fall back to the prep-less DifferentiationInterface
        # methods, which prepare internally on each call.
    DifferentiationInterface.pullback!(compute_adable, (dX_ca,), backend, X_ca, (dY_ca,))
    else
    DifferentiationInterface.pullback!(compute_adable, (dX_ca,), prep, backend, X_ca, (dY_ca,))
    end

    # Now copy the input derivatives to `d_inputs`:
    for iname in keys(dX_ca)
        iname_str = string(iname)
        if typeof(d_inputs[iname_str]) <: AbstractArray
            d_inputs[iname_str] .+= @view(dX_ca[iname])
        else
            d_inputs[iname_str] += only(dX_ca[iname])
        end
    end

    return nothing
end

"""
    compute_jacvec_product!(self::ADExplicitComp{<:MatrixFreeFlavor}, inputs, d_inputs, d_outputs, mode)

Compute the Jacobian-vector product (`mode="fwd"`, requires
[`MatrixFreeForwardFlavor`](@ref)) or vector-Jacobian product (`mode="rev"`,
requires [`MatrixFreeReverseFlavor`](@ref)).
"""
function OpenMDAOCore.compute_jacvec_product!(self::ADExplicitComp{<:MatrixFreeFlavor}, inputs, d_inputs, d_outputs, mode)
    backend = get_backend(self)
    prep = get_prep(self)
    if mode == "fwd"
        if self isa ADExplicitComp{MatrixFreeForwardFlavor}
            _compute_pushforward!(self, inputs, d_inputs, d_outputs)
        else
            @warn "mode = \"fwd\" not supported for AD backend $(nameof(typeof(backend))), preparation $(nameof(typeof(prep))), derivatives for $(nameof(typeof(self))) will be incorrect"
        end
    elseif mode == "rev"
        if self isa ADExplicitComp{MatrixFreeReverseFlavor}
            _compute_pullback!(self, inputs, d_inputs, d_outputs)
        else
            @warn "mode = \"rev\" not supported for AD backend $(nameof(typeof(backend))), preparation $(nameof(typeof(prep))), derivatives for $(nameof(typeof(self))) will be incorrect"
        end
    else
        throw(ArgumentError("unknown mode = \"$(mode)\""))
    end

    return nothing
end
