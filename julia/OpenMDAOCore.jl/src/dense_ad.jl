"""
    DenseADExplicitComp{InPlace,TAD,TCompute,TX,TY,TJ,TPrep,TXCS,TYCS,TAMD} <: AbstractExplicitComp{InPlace}

An `<:AbstractADExplicitComp` for dense Jacobians.

# Fields
* `ad_backend::TAD`: `<:ADTypes.AbstractADType` automatic differentation "backend" library
* `compute_adable::TCompute`: function of the form `compute_adable(Y, X)` or `Y = compute_adable(x)` compatible with DifferentiationInterface.jl that performs the desired computation, where `Y` and `X` are `ComponentVector`s of outputs and inputs, respectively
* `X_ca::ComponentVector`: `ComponentVector` of inputs
* `Y_ca::ComponentVector`: `ComponentVector` of outputs
* `J_ca::ComponentMatrix`: Dense `ComponentMatrix` of the Jacobian of `Y_ca` with respect to `X_ca`
* `units_dict::Dict{Symbol,String}`: mapping of variable names to units. Can be an empty `Dict` if units are not desired.
* `tags_dict::Dict{Symbol,Vector{String}`: mapping of variable names to `Vector`s of `String`s specifing variable tags.
* `shape_by_conn_dict::Dict{Symbol,Bool}`: mapping of variable names to `Bool` indicating if the variable shape should be determined dynamically by a connection.
* `copy_shape_dict::Dict{Symbol,Symbol}`: mapping of variable names to variable names indicating if a variable shape should be copied from another variable.
* `prep::DifferentiationInterface.JacobianPrep`: `DifferentiationInterface.jl` "preparation" object
* `X_ca::ComponentVector`: `ComplexF64` version of `X_ca` (for the complex-step method)
* `Y_ca::ComponentVector`: `ComplexF64` version of `Y_ca` (for the complex-step method)
"""
struct DenseADExplicitComp{InPlace,TAD,TCompute,TX,TY,TJ,TPrep,TXCS,TYCS} <: AbstractADExplicitComp{InPlace}
    ad_backend::TAD
    compute_adable::TCompute
    X_ca::TX
    Y_ca::TY
    J_ca::TJ
    prep::TPrep
    units_dict::Dict{Symbol,String}
    tags_dict::Dict{Symbol,Vector{String}}
    shape_by_conn_dict::Dict{Symbol,Bool}
    copy_shape_dict::Dict{Symbol,Symbol}
    X_ca_cs::TXCS
    Y_ca_cs::TYCS

    function DenseADExplicitComp{InPlace}(ad_backend, compute_adable, X_ca, Y_ca, J_ca, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs) where {InPlace}
        return new{InPlace, typeof(ad_backend), typeof(compute_adable), typeof(X_ca), typeof(Y_ca),
                   typeof(J_ca), typeof(prep),
                   typeof(X_ca_cs), typeof(Y_ca_cs)}(ad_backend,
                                             compute_adable, X_ca,
                                             Y_ca, J_ca,
                                             prep,
                                             units_dict,
                                             tags_dict,
                                             shape_by_conn_dict,
                                             copy_shape_dict,
                                             X_ca_cs, Y_ca_cs)
    end
end

function DenseADExplicitComp{false}(ad_backend, compute_adable, X_ca, J_ca, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs)
    Y_ca = nothing
    Y_ca_cs = nothing
    return DenseADExplicitComp{false}(ad_backend, compute_adable, X_ca, Y_ca, J_ca, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs)
end

"""
    DenseADExplicitComp(ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create a `DenseADExplicitComp` from a user-defined function and output and input `ComponentVector`s.

# Positional Arguments
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentation "backend" library
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
function DenseADExplicitComp(ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}

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
        prep, J_ca, X_ca_cs, Y_ca_cs = _get_dense_prep_stuff(ad_backend, compute_adable, Y_ca, X_ca)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        prep = J_ca = X_ca_cs = Y_ca_cs = nothing
    end

    return DenseADExplicitComp{true}(ad_backend, compute_adable, X_ca, Y_ca, J_ca, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs)
end

"""
    DenseADExplicitComp(ad_backend, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create a `DenseADExplicitComp` from a user-defined function and output and input `ComponentVector`s.

# Positional Arguments
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentation "backend" library
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
function DenseADExplicitComp(ad_backend::TAD, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}

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
        prep, J_ca, X_ca_cs = _get_dense_prep_stuff(ad_backend, compute_adable, X_ca)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        prep = J_ca = X_ca_cs = nothing
    end

    return DenseADExplicitComp{false}(ad_backend, compute_adable, X_ca, J_ca, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs)
end

get_jacobian_ca(comp::DenseADExplicitComp) = comp.J_ca

function _get_dense_prep_stuff(ad_backend, f!, Y_ca, X_ca)
    # Need to "prepare" the backend.
    prep = DifferentiationInterface.prepare_jacobian(f!, Y_ca, ad_backend, X_ca)

    # Get the Jacobian matrix.
    TF = promote_type(eltype(Y_ca), eltype(X_ca))
    J = Matrix{TF}(undef, length(Y_ca), length(X_ca))

    # Then use that Jacobian to create the component matrix version.
    J_ca = ComponentMatrix(J, (only(getaxes(Y_ca,)), only(getaxes(X_ca))))

    # Create complex-valued versions of the X_ca and Y_ca arrays.
    TCS = Complex{TF}
    X_ca_cs = similar(X_ca, TCS)
    Y_ca_cs = similar(Y_ca, TCS)

    return prep, J_ca, X_ca_cs, Y_ca_cs
end

function _get_dense_prep_stuff(ad_backend, f, X_ca)
    # Need to "prepare" the backend.
    prep = DifferentiationInterface.prepare_jacobian(f, ad_backend, X_ca)

    # Need the output component vector to define the axes of the Jacobian.
    Y_ca = f(X_ca)

    # Now I think I can get the sparse Jacobian from that.
    TF = promote_type(eltype(Y_ca), eltype(X_ca))
    J = Matrix{TF}(undef, length(Y_ca), length(X_ca))

    # Then use that sparse Jacobian to create the component matrix version.
    J_ca = ComponentMatrix(J, (only(getaxes(Y_ca,)), only(getaxes(X_ca))))

    # Create complex-valued versions of the X_ca_full and Y_ca_full arrays.
    X_ca_cs = similar(X_ca, ComplexF64)

    return prep, J_ca, X_ca_cs
end

function update_prep(self::DenseADExplicitComp{true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})

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
        prep, J_ca, X_ca_cs, Y_ca_cs = _get_dense_prep_stuff(ad_backend, f!, Y_ca, X_ca)

        # Now just copy things over.
        units_dict = self.units_dict
        tags_dict = self.tags_dict
        shape_by_conn_dict = self.shape_by_conn_dict
        copy_shape_dict = self.copy_shape_dict

        self = DenseADExplicitComp{true}(ad_backend, f!, X_ca, Y_ca, J_ca, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs)
    end

    return self
end

function update_prep(self::DenseADExplicitComp{false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})

    if length(input_sizes) > 0
        X_ca_old = get_input_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        # Get the new sparsity stuff.
        ad_backend = get_backend(self)
        f = get_callback(self)
        prep, J_ca, X_ca_cs = _get_dense_prep_stuff(ad_backend, f, X_ca)

        # Now just copy things over.
        units_dict = self.units_dict
        tags_dict = self.tags_dict
        shape_by_conn_dict = self.shape_by_conn_dict
        copy_shape_dict = self.copy_shape_dict

        self = DenseADExplicitComp{false}(ad_backend, f, X_ca, J_ca, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs)
    end

    return self
end

function get_partials_data(self::DenseADExplicitComp)
    return [OpenMDAOCore.PartialsData("*", "*")]
end

function setup_partials(self::DenseADExplicitComp, input_sizes, output_sizes)

    input_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in input_sizes)
    output_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in output_sizes)

    self_new = update_prep(self, input_sizes_ca, output_sizes_ca)

    # Now finally get the partials data.
    return self_new, get_partials_data(self_new)
end

function OpenMDAOCore.compute_partials!(self::DenseADExplicitComp{true}, inputs, partials)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    # Get the Jacobian.
    f! = get_callback(self)
    Y_ca = get_output_ca(self)
    J_ca = get_jacobian_ca(self)
    prep = get_prep(self)
    ad_backend = get_backend(self)
    DifferentiationInterface.jacobian!(f!, Y_ca, J_ca, prep, ad_backend, X_ca)

    # Extract the derivatives from `J_ca` and put them in `partials`.
    raxis, caxis = getaxes(J_ca)
    for oname in keys(raxis)
        for iname in keys(caxis)
            # Grab the subjacobian we're interested in.
            Jsub_in = @view(J_ca[oname, iname])

            # OpenMDAO might not ask for all the partials, and so all combination of output/input keys might not be present in `partials`.
            local Jsub_out
            try
                Jsub_out = partials[string(oname), string(iname)]
            catch e
                if !isa(e, KeyError)
                    rethrow()
                end
            else
                # Now we should be able to write the partials to the OpenMDAO Python array.
                Jsub_out .= Jsub_in
            end

        end
    end

    return nothing
end

function OpenMDAOCore.compute_partials!(self::DenseADExplicitComp{false}, inputs, partials)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    # Get the Jacobian.
    f = get_callback(self)
    J_ca = get_jacobian_ca(self)
    prep = get_prep(self)
    ad_backend = get_backend(self)
    DifferentiationInterface.jacobian!(f, J_ca, prep, ad_backend, X_ca)

    # Extract the derivatives from `J_ca` and put them in `partials`.
    raxis, caxis = getaxes(J_ca)
    for oname in keys(raxis)
        for iname in keys(caxis)
            # Grab the subjacobian we're interested in.
            Jsub_in = @view(J_ca[oname, iname])
            
            # OpenMDAO might not ask for all the partials, and so all combination of output/input keys might not be present in `partials`.
            local Jsub_out
            try
                Jsub_out = partials[string(oname), string(iname)]
            catch e
                if !isa(e, KeyError)
                    rethrow()
                end
            else
                # Now we should be able to write the partials to the OpenMDAO Python array.
                Jsub_out .= Jsub_in
            end
        end
    end

    return nothing
end

has_setup_partials(self::DenseADExplicitComp) = true
has_compute_partials(self::DenseADExplicitComp) = true
has_compute_jacvec_product(self::DenseADExplicitComp) = false
