// RUN: aster-opt %s --verify-roundtrip

func.func private @test_load(%arg0: !ptr.ptr<#gpu.address_space<global>>)
