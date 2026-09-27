@testitem "autodense, in-place, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseTestPrep(4, 3, "forwarddiff"))
end

@testitem "autodense, in-place, reversediff" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseTestPrep(4, 3, "reversediff"))
end

@testitem "autodense, in-place, enzymeforward" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseTestPrep(4, 3, "enzymeforward"))
end

@testitem "autodense, in-place, enzymereverse" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseTestPrep(4, 3, "enzymereverse"))
end

@testitem "autodense, in-place, shape_by_conn, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseShapeByConnTestPrep(4, 3, "forwarddiff"))
end

@testitem "autodense, in-place, shape_by_conn, reversediff" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseShapeByConnTestPrep(4, 3, "reversediff"))
end

@testitem "autodense, in-place, shape_by_conn, enzymeforward" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseShapeByConnTestPrep(4, 3, "enzymeforward"))
end

@testitem "autodense, in-place, shape_by_conn, enzymereverse" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseShapeByConnTestPrep(4, 3, "enzymereverse"))
end

@testitem "autodense, out-of-place, forwarddiff" setup=[ADCallbacks] begin
    doit_out_of_place(AutoDenseTestPrep(4, 3, "forwarddiff"))
end

@testitem "autodense, out-of-place, reversediff" setup=[ADCallbacks] begin
    doit_out_of_place(AutoDenseTestPrep(4, 3, "reversediff"))
end

@testitem "autodense, out-of-place, zygote" setup=[ADCallbacks] begin
    doit_out_of_place(AutoDenseTestPrep(4, 3, "zygote"))
end

@testitem "autodense, out-of-place, shape_by_conn, forwarddiff" setup=[ADCallbacks] begin
    doit_out_of_place(AutoDenseShapeByConnTestPrep(4, 3, "forwarddiff"))
end

@testitem "autodense, out-of-place, shape_by_conn, reversediff" setup=[ADCallbacks] begin
    doit_out_of_place(AutoDenseShapeByConnTestPrep(4, 3, "reversediff"))
end

@testitem "autodense, out-of-place, shape_by_conn, zygote" setup=[ADCallbacks] begin
    doit_out_of_place(AutoDenseShapeByConnTestPrep(4, 3, "zygote"))
end
