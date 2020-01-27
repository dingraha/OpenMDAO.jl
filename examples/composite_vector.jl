using OpenMDAO

function NamedCompositeVector(vars::Vector{VarData{T, N}}) where {T, N}
    ddata = Dict{String, Array{T, N}}()
    for var in vars
        ddata[var.name] = Array{T}(undef, var.shape...)
        @. ddata[var.name] = var.val
    end

    return NamedCompositeVector(ddata)
end

function NamedCompositeMatrix(inputs::Vector{VarData{T, N}}, outputs::Vector{VarData{T, N}}) where {T, N}
    ddata = Dict{Tuple{String, String}, Array{T, 2}}()
    for ivar in inputs
        for ovar in outputs
            ddata[(ovar.name, ivar.name)] = Array{T}(undef, prod(ovar.shape), prod(ivar.shape))
        end
    end
    return NamedCompositeMatrix(ddata)
end
