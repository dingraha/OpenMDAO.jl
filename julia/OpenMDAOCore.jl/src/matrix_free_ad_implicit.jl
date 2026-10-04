# Matrix-free-flavor implicit AD component: constructors and methods.
#
# Note: the prep builders from `matrix_free_ad.jl`
# (`_get_matrix_free_forward_prep_in_place`, etc.) are reused for implicit
# components: the implicit `compute_adable(R, YX)` closures already have
# DifferentiationInterface's in-place (`f!(y, x)`) or out-of-place
# (`y = f(x)`) forms, so we pass `R_ca` as the "output" argument (`Y_ca`) and
# the combined `YX_ca` as the "input" argument (`X_ca`). In the resulting
# `MatrixFreeDerivPrep`, `dX_ca` holds the combined dYX tangent/cotangent
# buffer and `dY_ca` holds the residual tangent/cotangent buffer dR.

"""
    ADImplicitComp(::MatrixFreeForwardFlavor, ::Val{true}, ad_backend, f!, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an in-place [`MatrixFreeForwardFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pushforward!`.
"""
function ADImplicitComp(::MatrixFreeForwardFlavor, ::Val{true}, ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    R_ca = similar(Y_ca)

    compute_adable = _make_implicit_compute_adable(true, f!, params, Y_range, X_range, Y_axes, X_axes)
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_matrix_free_forward_prep_in_place(
            ad_backend, compute_adable, R_ca, YX_ca, force_skip_prep)
    else
        # Shapes not yet known: defer to `update_prep` (called by OpenMDAO's
        # `setup_partials`). The prep-less DifferentiationInterface methods are
        # used at runtime until then.
        dYX_ca = ComponentVector{eltype(YX_ca)}()
        dR_ca = ComponentVector{eltype(R_ca)}()
        YX_ca_cs = ComponentVector{ComplexF64}()
        R_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dYX_ca, dR_ca, nothing)
    end

    return ADImplicitComp{MatrixFreeForwardFlavor, true}(ad_backend, compute_adable, f!, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes)
end

"""
    ADImplicitComp(::MatrixFreeForwardFlavor, ::Val{false}, ad_backend, f, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an out-of-place [`MatrixFreeForwardFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pushforward!`.
"""
function ADImplicitComp(::MatrixFreeForwardFlavor, ::Val{false}, ad_backend::TAD, f, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    compute_adable = _make_implicit_compute_adable(false, f, params, Y_range, X_range, Y_axes, X_axes)
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, YX_ca_cs = _get_matrix_free_forward_prep_out_of_place(
            ad_backend, compute_adable, compute_adable(YX_ca), YX_ca, force_skip_prep)
    else
        # Shapes not yet known: defer to `update_prep` (called by OpenMDAO's
        # `setup_partials`). The prep-less DifferentiationInterface methods are
        # used at runtime until then.
        dYX_ca = ComponentVector{eltype(YX_ca)}()
        dR_ca = ComponentVector{eltype(R_ca)}()
        YX_ca_cs = ComponentVector{ComplexF64}()
        R_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dYX_ca, dR_ca, nothing)
    end

    R_ca = R_ca_cs = nothing

    return ADImplicitComp{MatrixFreeForwardFlavor, false}(ad_backend, compute_adable, f, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes)
end

"""
    ADImplicitComp(::MatrixFreeReverseFlavor, ::Val{true}, ad_backend, f!, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an in-place [`MatrixFreeReverseFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pullback!`.
"""
function ADImplicitComp(::MatrixFreeReverseFlavor, ::Val{true}, ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    R_ca = similar(Y_ca)

    compute_adable = _make_implicit_compute_adable(true, f!, params, Y_range, X_range, Y_axes, X_axes)
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_matrix_free_reverse_prep_in_place(
            ad_backend, compute_adable, R_ca, YX_ca, force_skip_prep)
    else
        # Shapes not yet known: defer to `update_prep` (called by OpenMDAO's
        # `setup_partials`). The prep-less DifferentiationInterface methods are
        # used at runtime until then.
        dYX_ca = ComponentVector{eltype(YX_ca)}()
        dR_ca = ComponentVector{eltype(R_ca)}()
        YX_ca_cs = ComponentVector{ComplexF64}()
        R_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dYX_ca, dR_ca, nothing)
    end

    return ADImplicitComp{MatrixFreeReverseFlavor, true}(ad_backend, compute_adable, f!, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes)
end

"""
    ADImplicitComp(::MatrixFreeReverseFlavor, ::Val{false}, ad_backend, f, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an out-of-place [`MatrixFreeReverseFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pullback!`.
"""
function ADImplicitComp(::MatrixFreeReverseFlavor, ::Val{false}, ad_backend::TAD, f, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    compute_adable = _make_implicit_compute_adable(false, f, params, Y_range, X_range, Y_axes, X_axes)
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, YX_ca_cs = _get_matrix_free_reverse_prep_out_of_place(
            ad_backend, compute_adable, compute_adable(YX_ca), YX_ca, force_skip_prep)
    else
        # Shapes not yet known: defer to `update_prep` (called by OpenMDAO's
        # `setup_partials`). The prep-less DifferentiationInterface methods are
        # used at runtime until then.
        dYX_ca = ComponentVector{eltype(YX_ca)}()
        dR_ca = ComponentVector{eltype(R_ca)}()
        YX_ca_cs = ComponentVector{ComplexF64}()
        R_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dYX_ca, dR_ca, nothing)
    end

    R_ca = R_ca_cs = nothing

    return ADImplicitComp{MatrixFreeReverseFlavor, false}(ad_backend, compute_adable, f, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes)
end

function update_prep(comp::ADImplicitComp{MatrixFreeForwardFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        Y_ca_old = get_output_ca(comp)
        X_ca_old = get_input_ca(comp)

        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
        Y_range = 1:length(Y_ca)
        X_range = length(Y_ca)+1:length(YX_ca)
        Y_axes = getaxes(Y_ca)
        X_axes = getaxes(X_ca)

        compute_adable = _make_implicit_compute_adable(true, comp.func, comp.params, Y_range, X_range, Y_axes, X_axes)
        ad_backend = get_backend(comp)
        R_ca = similar(Y_ca)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_matrix_free_forward_prep_in_place(ad_backend, compute_adable, R_ca, YX_ca, false)

        comp = ADImplicitComp{MatrixFreeForwardFlavor, true}(get_backend(comp), compute_adable, comp.func, comp.params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes)
    end

    return comp
end

function _compute_implicit_pushforward!(comp::ADImplicitComp{MatrixFreeForwardFlavor, true}, inputs, outputs, dinputs, doutputs, dresids)
    YX_ca = get_combined_ca(comp)
    for uname in output_keys(comp)
        @view(YX_ca[uname]) .= outputs[string(uname)]
    end
    for iname in input_keys(comp)
        @view(YX_ca[iname]) .= inputs[string(iname)]
    end

    # Assemble the YX tangent from doutputs (Y part) and dinputs (X part).
    dYX_ca = get_dcombined_ca(comp)
    for uname in output_keys(comp)
        ustr = string(uname)
        @view(dYX_ca[uname]) .= doutputs[ustr]
    end
    for iname in input_keys(comp)
        istr = string(iname)
        @view(dYX_ca[iname]) .= dinputs[istr]
    end

    R_ca = get_residual_ca(comp)
    dR_ca = get_dresidual_ca(comp)
    compute_adable = get_callback(comp)
    prep = get_prep(comp)
    backend = get_backend(comp)

    if prep === nothing
        isempty(dYX_ca) && error("The DifferentiationInterface prep for this component has not been created. This happens when the component is created with `shape_by_conn`/`copy_shape` variables and `update_prep` has not been called yet (OpenMDAO does this automatically during problem setup).")
        # No prep available (e.g. `force_skip_prep=true`): fall back to the
        # prep-less DifferentiationInterface methods, which prepare internally
        # on each call.
    DifferentiationInterface.pushforward!(compute_adable, R_ca, (dR_ca,), backend, YX_ca, (dYX_ca,))
    else
    DifferentiationInterface.pushforward!(compute_adable, R_ca, (dR_ca,), prep, backend, YX_ca, (dYX_ca,))
    end

    for rname in keys(dR_ca)
        rstr = string(rname)
        if typeof(dresids[rstr]) <: AbstractArray
            dresids[rstr] .+= @view(dR_ca[rname])
        else
            dresids[rstr] += only(dR_ca[rname])
        end
    end

    return nothing
end

function update_prep(comp::ADImplicitComp{MatrixFreeForwardFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        Y_ca_old = get_output_ca(comp)
        X_ca_old = get_input_ca(comp)

        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
        Y_range = 1:length(Y_ca)
        X_range = length(Y_ca)+1:length(YX_ca)
        Y_axes = getaxes(Y_ca)
        X_axes = getaxes(X_ca)

        compute_adable = _make_implicit_compute_adable(false, comp.func, comp.params, Y_range, X_range, Y_axes, X_axes)
        ad_backend = get_backend(comp)
        R_ca = nothing
        R_ca_cs = nothing
        deriv_prep, YX_ca_cs = _get_matrix_free_forward_prep_out_of_place(ad_backend, compute_adable, YX_ca, false)

        comp = ADImplicitComp{MatrixFreeForwardFlavor, false}(get_backend(comp), compute_adable, comp.func, comp.params, YX_ca, nothing, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes)
    end

    return comp
end

function _compute_implicit_pushforward!(comp::ADImplicitComp{MatrixFreeForwardFlavor, false}, inputs, outputs, dinputs, doutputs, dresids)
    YX_ca = get_combined_ca(comp)
    for uname in output_keys(comp)
        @view(YX_ca[uname]) .= outputs[string(uname)]
    end
    for iname in input_keys(comp)
        @view(YX_ca[iname]) .= inputs[string(iname)]
    end

    dYX_ca = get_dcombined_ca(comp)
    for uname in output_keys(comp)
        ustr = string(uname)
        @view(dYX_ca[uname]) .= doutputs[ustr]
    end
    for iname in input_keys(comp)
        istr = string(iname)
        @view(dYX_ca[iname]) .= dinputs[istr]
    end

    dR_ca = get_dresidual_ca(comp)
    compute_adable = get_callback(comp)
    prep = get_prep(comp)
    backend = get_backend(comp)

    if prep === nothing
        isempty(dYX_ca) && error("The DifferentiationInterface prep for this component has not been created. This happens when the component is created with `shape_by_conn`/`copy_shape` variables and `update_prep` has not been called yet (OpenMDAO does this automatically during problem setup).")
        # No prep available (e.g. `force_skip_prep=true`): fall back to the
        # prep-less DifferentiationInterface methods, which prepare internally
        # on each call.
    DifferentiationInterface.pushforward!(compute_adable, (dR_ca,), backend, YX_ca, (dYX_ca,))
    else
    DifferentiationInterface.pushforward!(compute_adable, (dR_ca,), prep, backend, YX_ca, (dYX_ca,))
    end

    for rname in keys(dR_ca)
        rstr = string(rname)
        if typeof(dresids[rstr]) <: AbstractArray
            dresids[rstr] .+= @view(dR_ca[rname])
        else
            dresids[rstr] += only(dR_ca[rname])
        end
    end

    return nothing
end

function update_prep(comp::ADImplicitComp{MatrixFreeReverseFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        Y_ca_old = get_output_ca(comp)
        X_ca_old = get_input_ca(comp)

        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
        Y_range = 1:length(Y_ca)
        X_range = length(Y_ca)+1:length(YX_ca)
        Y_axes = getaxes(Y_ca)
        X_axes = getaxes(X_ca)

        compute_adable = _make_implicit_compute_adable(true, comp.func, comp.params, Y_range, X_range, Y_axes, X_axes)
        ad_backend = get_backend(comp)
        R_ca = similar(Y_ca)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_matrix_free_reverse_prep_in_place(ad_backend, compute_adable, R_ca, YX_ca, false)

        comp = ADImplicitComp{MatrixFreeReverseFlavor, true}(get_backend(comp), compute_adable, comp.func, comp.params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes)
    end

    return comp
end

function _compute_implicit_pullback!(comp::ADImplicitComp{MatrixFreeReverseFlavor, true}, inputs, outputs, dinputs, doutputs, dresids)
    YX_ca = get_combined_ca(comp)
    for uname in output_keys(comp)
        @view(YX_ca[uname]) .= outputs[string(uname)]
    end
    for iname in input_keys(comp)
        @view(YX_ca[iname]) .= inputs[string(iname)]
    end

    # Load the residual cotangent seed from dresids.
    dR_ca = get_dresidual_ca(comp)
    for rname in keys(dR_ca)
        rstr = string(rname)
        @view(dR_ca[rname]) .= dresids[rstr]
    end

    R_ca = get_residual_ca(comp)
    dYX_ca = get_dcombined_ca(comp)
    compute_adable = get_callback(comp)
    backend = get_backend(comp)
    prep = get_prep(comp)

    if prep === nothing
        isempty(dYX_ca) && error("The DifferentiationInterface prep for this component has not been created. This happens when the component is created with `shape_by_conn`/`copy_shape` variables and `update_prep` has not been called yet (OpenMDAO does this automatically during problem setup).")
        # No prep available (e.g. `force_skip_prep=true`): fall back to the
        # prep-less DifferentiationInterface methods, which prepare internally
        # on each call.
    DifferentiationInterface.pullback!(compute_adable, R_ca, (dYX_ca,), backend, YX_ca, (dR_ca,))
    else
    DifferentiationInterface.pullback!(compute_adable, R_ca, (dYX_ca,), prep, backend, YX_ca, (dR_ca,))
    end

    # Scatter dYX cotangents back: Y part → doutputs, X part → dinputs.
    for uname in output_keys(comp)
        ustr = string(uname)
        if typeof(doutputs[ustr]) <: AbstractArray
            doutputs[ustr] .+= @view(dYX_ca[uname])
        else
            doutputs[ustr] += only(dYX_ca[uname])
        end
    end
    for iname in input_keys(comp)
        istr = string(iname)
        if typeof(dinputs[istr]) <: AbstractArray
            dinputs[istr] .+= @view(dYX_ca[iname])
        else
            dinputs[istr] += only(dYX_ca[iname])
        end
    end

    return nothing
end

function update_prep(comp::ADImplicitComp{MatrixFreeReverseFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        Y_ca_old = get_output_ca(comp)
        X_ca_old = get_input_ca(comp)

        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
        Y_range = 1:length(Y_ca)
        X_range = length(Y_ca)+1:length(YX_ca)
        Y_axes = getaxes(Y_ca)
        X_axes = getaxes(X_ca)

        compute_adable = _make_implicit_compute_adable(false, comp.func, comp.params, Y_range, X_range, Y_axes, X_axes)
        ad_backend = get_backend(comp)
        R_ca = nothing
        R_ca_cs = nothing
        deriv_prep, YX_ca_cs = _get_matrix_free_reverse_prep_out_of_place(ad_backend, compute_adable, YX_ca, false)

        comp = ADImplicitComp{MatrixFreeReverseFlavor, false}(get_backend(comp), compute_adable, comp.func, comp.params, YX_ca, nothing, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes)
    end

    return comp
end

function _compute_implicit_pullback!(comp::ADImplicitComp{MatrixFreeReverseFlavor, false}, inputs, outputs, dinputs, doutputs, dresids)
    YX_ca = get_combined_ca(comp)
    for uname in output_keys(comp)
        @view(YX_ca[uname]) .= outputs[string(uname)]
    end
    for iname in input_keys(comp)
        @view(YX_ca[iname]) .= inputs[string(iname)]
    end

    dR_ca = get_dresidual_ca(comp)
    for rname in keys(dR_ca)
        rstr = string(rname)
        @view(dR_ca[rname]) .= dresids[rstr]
    end

    dYX_ca = get_dcombined_ca(comp)
    compute_adable = get_callback(comp)
    backend = get_backend(comp)
    prep = get_prep(comp)

    if prep === nothing
        isempty(dYX_ca) && error("The DifferentiationInterface prep for this component has not been created. This happens when the component is created with `shape_by_conn`/`copy_shape` variables and `update_prep` has not been called yet (OpenMDAO does this automatically during problem setup).")
        # No prep available (e.g. `force_skip_prep=true`): fall back to the
        # prep-less DifferentiationInterface methods, which prepare internally
        # on each call.
    DifferentiationInterface.pullback!(compute_adable, (dYX_ca,), backend, YX_ca, (dR_ca,))
    else
    DifferentiationInterface.pullback!(compute_adable, (dYX_ca,), prep, backend, YX_ca, (dR_ca,))
    end

    for uname in output_keys(comp)
        ustr = string(uname)
        if typeof(doutputs[ustr]) <: AbstractArray
            doutputs[ustr] .+= @view(dYX_ca[uname])
        else
            doutputs[ustr] += only(dYX_ca[uname])
        end
    end
    for iname in input_keys(comp)
        istr = string(iname)
        if typeof(dinputs[istr]) <: AbstractArray
            dinputs[istr] .+= @view(dYX_ca[iname])
        else
            dinputs[istr] += only(dYX_ca[iname])
        end
    end

    return nothing
end

function apply_linear!(comp::ADImplicitComp{MatrixFreeForwardFlavor}, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
    if mode == "fwd"
        _compute_implicit_pushforward!(comp, inputs, outputs, d_inputs, d_outputs, d_residuals)
    else
        throw(ArgumentError("MatrixFreeForwardFlavor only supports mode=\"fwd\""))
    end
    return nothing
end

function apply_linear!(comp::ADImplicitComp{MatrixFreeReverseFlavor}, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
    if mode == "rev"
        _compute_implicit_pullback!(comp, inputs, outputs, d_inputs, d_outputs, d_residuals)
    else
        throw(ArgumentError("MatrixFreeReverseFlavor only supports mode=\"rev\""))
    end
    return nothing
end
