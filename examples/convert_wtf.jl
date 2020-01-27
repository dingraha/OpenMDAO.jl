import Base.convert

abstract type AbstractFoo end

mutable struct Foo <: AbstractFoo
    a
end

struct Bar
    b
end

function convert(::Type{T}, po::Bar) where {T<:AbstractFoo}
    println("In AbstractFoo's convert")
    return T(po.b)
end

function convert(::Type{Foo}, po::Bar)
    println("In Foo's convert")
    return Foo(po.b+1)
end

bar = Bar(8)
@show bar
foo = convert(Foo, bar)
@show foo
