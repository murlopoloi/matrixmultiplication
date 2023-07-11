### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 8122ac40-1ab5-11ee-351f-a9cd2caf3adc
using LinearAlgebra

# ╔═╡ 4b1ab40e-b619-4eb5-bd0b-1b2e4334484d
using BenchmarkTools

# ╔═╡ d7f6cb54-f937-4254-a26c-37d331464af5
function basemult(A, B, C)

	# Naive multiplication algorithm with
	# C as input to optimize memory
	# allocation.
	
    n, m = size(A)
    p = size(B)[2]

    for i = 1:n
        for j = 1:p
            soma = 0
            for k = 1:m
                soma += A[i, k]*B[k, j]
            end
            C[i,j] = soma
        end
    end

    return C
end

# ╔═╡ 80bee335-af98-47e5-8797-d03846acfe42
function matfill(A, B)
	
	# Pads the matrices A and B.
	
    k, l = size(A)
    m, n = size(B)

    max_A = max(k, l)
    max_B = max(m, n)
    max_size = max(max_A, max_B)
    flag = 0
    
    while max_size > (2^flag)
        flag = flag + 1
    end

    new_A = zeros(2^flag, 2^flag) 
    new_B = zeros(2^flag, 2^flag)
    new_A[1:k, 1:l] = A
    new_B[1:m, 1:n] = B

    return new_A, new_B
end

# ╔═╡ 85b69611-ab2d-470c-aa01-e9000c5be671
function sizecheck(A,B)
	
	# Checks if the matrices sizes are appropiate 
	# for computing their product. 
	
    k, l = size(A)
    m, n = size(B)
    if (k == l && m == n) == false
        return false
    else 
        return true
    end
end

# ╔═╡ fbd163dc-f810-47fc-8ea2-7e61fef0dbc3
function ipwin(A,B,C)

	# In-Place Winograd algorithm.
	
	n = size(A,1)
	
	if n == 1
		C[1,1] = A[1,1]*B[1,1]
		return C
	end
	
	k = div(n,2)

	A11 = view(A, 1:k, 1:k) 
	A12 = view(A, 1:k, k+1:n)
	A21 = view(A, k+1:n, 1:k) 
	A22 = view(A, k+1:n, k+1:n)
	
	B11 = view(B, 1:k, 1:k)
	B12 = view(B, 1:k, k+1:n)
	B21 = view(B, k+1:n, 1:k)
	B22 = view(B, k+1:n, k+1:n)
	
	C11 = view(C, 1:k, 1:k)
	C12 = view(C, 1:k, k+1:n)
	C21 = view(C, k+1:n, 1:k) 
	C22 = view(C, k+1:n, k+1:n)
	
	copy!(C11, A11)
	axpby!(-1, A21, 1, C11) #1 - S3 = A11 - A21 (C11)

	axpby!(1, A22, 1, A21) #2 - S1 = A21 + A22 (A21)

	copy!(C22, B12)
	axpby!(-1, B11, 1, C22) #3 - T1 = B12 - B11 (C22)

	axpby!(1, B22, -1, B12) #4 - T3 = B22 - B12 (B12)

	ipwin(C11, B12, C21) #5 - P7 = S3×T3 (C21)

	copy!(C12, A11)
	axpby!(1, A21, -1, C12) #6 - S2 = S1 - A11 (C12)

	ipwin(A11, B11, C11) #7 - P1 = A11×B11 (C11)

	copy!(B11, C22)
	axpby!(1, B22, -1, B11) #8 - T2 = B22 - T1 (B11)

	ipwin(A21, C22, A11) #9 - P5 = S1×T1 (A11)

	copy!(C22, B21)
	axpby!(1, B11, -1, C22) #10 - T3 = T2 - B21 (C22)

	ipwin(A22, C22, A21) #11 - P4 = A22×T4 (A21)

	copy!(A22, C12)
	axpby!(1, A12, -1, A22) #12 - S4 = A12 - S2 (A22)

	ipwin(C12, B11, C22) #13 - P6 = S2×T2 (C22)

	axpby!(1, C11, 1, C22) #14 - U2 = P1 + P6 (C22)

	ipwin(A12, B21, C12) #15 - P2 = A12×B21 (C12)

	axpby!(1, C12, 1, C11) #16 - U1 = P1 + P2 (C11)

	copy!(C12, A11)
	axpby!(1, C22, 1, C12) #17 - U4 = U2 + P5 (C12)

	axpby!(1, C21, 1, C22) #18 - U3 = U2 + P7 (C22)

	copy!(C21, A21)
	axpby!(1, C22 , -1, C21) # 19 - U6 = U3 - P4 (C21)

	axpby!(1, A11, 1, C22) #20 U7 = U3 + P5 (C22)

	ipwin(A22, B22, A12) #21 P3 = S4×B22 (A12)

	axpby!(1, A12, 1, C12) #22 U5 = U4 + P3 (C12)
	
	return C #[U1 U5; U6 U7]
end



# ╔═╡ 2e1a84be-5cce-47ee-95e6-cd3116ed34be
function ipwinmod(A,B,C)

	# In-Place Winograd algorithm with 
	# n ≤ 12 modification.
	
	n = size(A,1)
	
	if n <= 12
		return basemult(A,B,C)
	end
	
	k = div(n,2)

	A11 = view(A, 1:k, 1:k) 
	A12 = view(A, 1:k, k+1:n)
	A21 = view(A, k+1:n, 1:k) 
	A22 = view(A, k+1:n, k+1:n)
	
	B11 = view(B, 1:k, 1:k)
	B12 = view(B, 1:k, k+1:n)
	B21 = view(B, k+1:n, 1:k)
	B22 = view(B, k+1:n, k+1:n)
	
	C11 = view(C, 1:k, 1:k)
	C12 = view(C, 1:k, k+1:n)
	C21 = view(C, k+1:n, 1:k) 
	C22 = view(C, k+1:n, k+1:n)
	
	copy!(C11, A11)
	axpby!(-1, A21, 1, C11) #1 - S3 = A11 - A21 (C11)

	axpby!(1, A22, 1, A21) #2 - S1 = A21 + A22 (A21)

	copy!(C22, B12)
	axpby!(-1, B11, 1, C22) #3 - T1 = B12 - B11 (C22)

	axpby!(1, B22, -1, B12) #4 - T3 = B22 - B12 (B12)

	ipwinmod(C11, B12, C21) #5 - P7 = S3×T3 (C21)

	copy!(C12, A11)
	axpby!(1, A21, -1, C12) #6 - S2 = S1 - A11 (C12)

	ipwinmod(A11, B11, C11) #7 - P1 = A11×B11 (C11)

	copy!(B11, C22)
	axpby!(1, B22, -1, B11) #8 - T2 = B22 - T1 (B11)

	ipwinmod(A21, C22, A11) #9 - P5 = S1×T1 (A11)

	copy!(C22, B21)
	axpby!(1, B11, -1, C22) #10 - T3 = T2 - B21 (C22)

	ipwinmod(A22, C22, A21) #11 - P4 = A22×T4 (A21)

	copy!(A22, C12)
	axpby!(1, A12, -1, A22) #12 - S4 = A12 - S2 (A22)

	ipwinmod(C12, B11, C22) #13 - P6 = S2×T2 (C22)

	axpby!(1, C11, 1, C22) #14 - U2 = P1 + P6 (C22)

	ipwinmod(A12, B21, C12) #15 - P2 = A12×B21 (C12)

	axpby!(1, C12, 1, C11) #16 - U1 = P1 + P2 (C11)

	copy!(C12, A11)
	axpby!(1, C22, 1, C12) #17 - U4 = U2 + P5 (C12)

	axpby!(1, C21, 1, C22) #18 - U3 = U2 + P7 (C22)

	copy!(C21, A21)
	axpby!(1, C22 , -1, C21) # 19 - U6 = U3 - P4 (C21)

	axpby!(1, A11, 1, C22) #20 U7 = U3 + P5 (C22)

	ipwinmod(A22, B22, A12) #21 P3 = S4×B22 (A12)

	axpby!(1, A12, 1, C12) #22 U5 = U4 + P3 (C12)
	
	return C #[U1 U5; U6 U7]
end



# ╔═╡ cf651483-3f92-485b-9033-d6cde2b28ffd
function optipwin(A, B)

	# Runs the ipwin function and checks
	# if padding is necessary.
	
	og_m, og_n = size(A)[1], size(B)[2]
    if sizecheck(A, B) == false 
        A, B = matfill(A, B)
    end
	n = size(A)[1]
	C = zeros(eltype(A),n,n) 
	return ipwin(copy(A),copy(B),C)[1:og_m, 1:og_n]
end


# ╔═╡ 80d336d9-16f5-492c-a208-3546fdb2a299
function optipwinmod(A, B)

	# Runs the ipwinmod function and checks
	# if padding is necessary.
	
	og_m, og_n = size(A)[1], size(B)[2]
    if sizecheck(A, B) == false 
        A, B = matfill(A, B)
    end
	n = size(A)[1]
	C = zeros(eltype(A),n,n) 
	return ipwinmod(copy(A),copy(B),C)[1:og_m, 1:og_n]
end


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[compat]
BenchmarkTools = "~1.3.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "117e7caf859d7b7be48d5fa00a6c0f726447b418"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "4b2e829ee66d4218e0cef22c0a64ee37cf258c29"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═8122ac40-1ab5-11ee-351f-a9cd2caf3adc
# ╠═4b1ab40e-b619-4eb5-bd0b-1b2e4334484d
# ╟─d7f6cb54-f937-4254-a26c-37d331464af5
# ╟─80bee335-af98-47e5-8797-d03846acfe42
# ╟─85b69611-ab2d-470c-aa01-e9000c5be671
# ╟─fbd163dc-f810-47fc-8ea2-7e61fef0dbc3
# ╟─2e1a84be-5cce-47ee-95e6-cd3116ed34be
# ╠═cf651483-3f92-485b-9033-d6cde2b28ffd
# ╠═80d336d9-16f5-492c-a208-3546fdb2a299
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
