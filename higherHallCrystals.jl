# functions for scaling dimensions in a coupled wire construction

module higherHall
using SparseArrays, LinearAlgebra, SpecialFunctions, QuadGK, StatsBase
export find_wire_positions_and_velocities, make_U, make_V, make_K, make_M, special_orthogonal_diagonalization, all_possible_ops
export Δ, get_chern, leading_instabilities, U_of_x_y_quadgk, find_maxU, density_density_couplings, fix_pixels, F_of_x_y_z_quadgk

function Δ(m_ops::Vector{Vector{Int64}},M::Matrix{Float64})::Vector{Float64}
    # returns scaling dimension of m_op given a scaling dimension matrix M 
    # Inputs: `m_op`` ~ a vector of vectors which are to be padded with zeros. M ~ a matrix of size >= length(m_op[1])
    # Outputs: a list of scaling dimensions Δ = (1/2) m^T M m for each m ∈ m_op

    dim = size(M,1)
    scdms = zeros(Float64,length(m_ops))
    K = make_K(dim)
    for (i,op) in enumerate(m_ops)
        l = length(op)
        op_padded = K * vcat(op,zeros(dim-l)) # need this! 7/20/24
        scdms[i] = 0.5*op_padded' * M * op_padded
    end
    return scdms 
end

function find_wire_positions_and_velocities(x,y::Vector{Float64}, nu::Float64)::Tuple{Vector{Float64},Vector{Float64}}
    ys = sort(y)
    @assert(length(y) == length(x))
    i1 = clamp(Int(round(nu * length(y))), 1, length(y) - 1)
    EF = (ys[i1] + ys[i1+1]) / 2 # this takes care of EF = y[i] edge cases
    
    wire_positions = Float64[]
    wire_velocities = Float64[]
    last_position = -Inf
    tol = 1e-4 # omit cases of double-counting 2 positions by omitting duplicates within tol of each other
    
    for i in 1:(length(y) - 1)
        if (y[i] - EF) * (y[i+1] - EF) <= 0 # this defines an intersection point
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
            new_position = x0
            if abs(new_position - last_position) > tol
                push!(wire_positions, new_position)
                push!(wire_velocities, (y1 - y0) / (x1 - x0))
                last_position = new_position
            end
        end
    end

    @assert((length(wire_positions) % 2) == 0) # even number of wires / u.c.

    if any(velocity -> abs(velocity) < 0.1, wire_velocities) # This defines a "minimum" (dim'less) velocity = 0.1 
        error("Velocity too low")
    end
    return wire_positions, wire_velocities
end

function find_maxU(x,y::Vector{Float64},nu_min::Float64,nu_max::Float64,am_min::Float64,am_max::Float64,coarse_mesh::Int,lB::Float64,λ::Float64)::Float64
    nu_vec = LinRange(nu_min, nu_max, coarse_mesh)
    am_vec = LinRange(am_min, am_max, coarse_mesh)
    current_max_U_over_v = 0.0 
    current_max_U = 0.0 
    r=0.0
    for nu in nu_vec
        for am in am_vec 
            wire_positions, wire_velocities = find_wire_positions_and_velocities(x,y,nu)
            vmin = minimum(abs.(wire_velocities)) 
            U = make_U(20,wire_positions,am,lB,λ,true)
            Umax = maximum(U)
            r = abs(Umax/vmin)
            if r > current_max_U_over_v
                current_max_U_over_v = r
                current_max_U = Umax
            end
        end
    end
    return current_max_U
end

function density_density_couplings(N::Int,wire_positions::Vector{Float64},aₘ::Float64,lB::Float64,λ::Float64,inversion_symmetric::Bool)::Tuple{Vector{Float64},Vector{Float64}}
    # Inputs. `N` ~ number of unit cells. `wire_positions` ~ vector of numbers inside [0,1] indicating the positions of wires inside unit cell
    # `aₘ` ~ moiré period, `lB` ~ magnetic length, `λ` ~ Coulomb screening length, `inversion_symmetric` ~ true iff the unit cell has an inversion symmetry about 0.5, then we save some time
    # Output is basically first row of `U`, capturing density-density interactions with an *arbitrary overall normalization*
    
    num_wires = length(wire_positions) # number of wires per unit cell
    dim = num_wires * N # U will be dim x dim
    Lx = aₘ*N # system size in X dir.
    UIJ = zeros(Float64, dim, dim) # initialize output 
    Δx_over_lB_IJ = zeros(Float64, dim, dim) # initialize output 
    # BACKEND FUNCTIONS

    function calculate_UIJ_and_Δx_over_lB_element(n::Int, J::Int, wire_positions::Vector{Float64}, num_wires::Int, aₘ::Float64, lB::Float64, Lx::Float64, λ::Float64)::Tuple{Float64,Float64}
        x1 = aₘ * wire_positions[n]
        unit_cells_apart = div(J-1,num_wires) # that's how many u.c.s over it is 
        J_in_fundamental_cell = J - num_wires*unit_cells_apart 
        x2 = aₘ * (unit_cells_apart + wire_positions[J_in_fundamental_cell])
        Δx_over_lB = (1/lB)* min(abs(x2 - x1), Lx - abs(x2 - x1)) # wrap around X direction
        val = U_of_x_y_quadgk(Δx_over_lB, lB / λ)
        if isnan(val)
            return Δx_over_lB,0.0
        else
            return Δx_over_lB,val
        end
    end

    function fill_UIJ!(UIJ::Matrix{Float64}, Δx_over_lB_IJ::Matrix{Float64},num_wires::Int, dim::Int, wire_positions::Vector{Float64}, aₘ::Float64, lB::Float64, Lx::Float64, λ::Float64, inversion_symmetric::Bool)
        for n in 1:num_wires
            if inversion_symmetric && n > div(num_wires, 2)
                row_idx = num_wires - n + 1 # this is the inversion-related row
                vec0 = UIJ[row_idx, :]
                vec0a= Δx_over_lB_IJ[row_idx, :]
                vec1 = circshift(vec0, -row_idx)
                vec1a = circshift(vec0a, -row_idx)
                vec3 = circshift(reverse(vec1), n-1)
                vec3a = circshift(reverse(vec1a), n-1)
                UIJ[n, :] = vec3
                Δx_over_lB_IJ[n, :] = vec3a
            else
                for J in 1:dim
                    Δx_over_lB, val = calculate_UIJ_and_Δx_over_lB_element(n, J, wire_positions, num_wires, aₘ, lB, Lx, λ)
                    UIJ[n, J] = val
                    Δx_over_lB_IJ[n, J] = Δx_over_lB
                end
            end
        end
    end

    # MAIN FUNCTION 

    # Populate first num_wires rows of matrix 
    fill_UIJ!(UIJ, Δx_over_lB_IJ,num_wires, dim, wire_positions, aₘ, lB, Lx, λ, inversion_symmetric)

    # Populate the rest using the first rows
    for n in 1:num_wires
        for I in 2:N
            row_idx = num_wires * (I - 1) + n
            UIJ[row_idx, :] = circshift(UIJ[n, :], row_idx - n)
            Δx_over_lB_IJ[row_idx, :] = circshift(Δx_over_lB_IJ[n, :], row_idx - n)
        end
    end

    if norm(UIJ-UIJ')<1e-7 # if it's basically symmetric
        UIJ = (UIJ+UIJ')/2; # make it exactly symmetric
    end
    @assert(issymmetric(UIJ))
    return UIJ[N,:], Δx_over_lB_IJ[N,:]
end

function make_U(N::Int,wire_positions::Vector{Float64},aₘ::Float64,lB::Float64,λ::Float64,inversion_symmetric::Bool)::Matrix{Float64}
    # Inputs. `N` ~ number of unit cells. `wire_positions` ~ vector of numbers inside [0,1] indicating the positions of wires inside unit cell
    # `aₘ` ~ moiré period, `lB` ~ magnetic length, `λ` ~ Coulomb screening length, `inversion_symmetric` ~ true iff the unit cell has an inversion symmetry about 0.5, then we save some time
    # Output. `U` ~ Matrix{Float64} capturing density-density interactions with an *arbitrary overall normalization*
    num_wires = length(wire_positions) # number of wires per unit cell
    dim = num_wires * N # U will be dim x dim
    Lx = aₘ*N # system size in X dir.
    UIJ = zeros(Float64, dim, dim) # initialize output 
    # BACKEND FUNCTIONS

    function calculate_UIJ_element(n::Int, J::Int, wire_positions::Vector{Float64}, num_wires::Int, aₘ::Float64, lB::Float64, Lx::Float64, λ::Float64)::Float64
        x1 = aₘ * wire_positions[n]
        unit_cells_apart = div(J-1,num_wires) # that's how many u.c.s over it is 
        J_in_fundamental_cell = J - num_wires*unit_cells_apart 
        x2 = aₘ * (unit_cells_apart + wire_positions[J_in_fundamental_cell])
        Δx_over_lB = (1/lB)* min(abs(x2 - x1), Lx - abs(x2 - x1)) # wrap around X direction
        val = U_of_x_y_quadgk(Δx_over_lB, lB / λ)
        if isnan(val)
            return 0.0
        else
            return val 
        end
    end

    function fill_UIJ!(UIJ::Matrix{Float64}, num_wires::Int, dim::Int, wire_positions::Vector{Float64}, aₘ::Float64, lB::Float64, Lx::Float64, λ::Float64, inversion_symmetric::Bool)
        for n in 1:num_wires
            if inversion_symmetric && n > div(num_wires, 2)
                row_idx = num_wires - n + 1 # this is the inversion-related row
                vec0 = UIJ[row_idx, :]
                vec1 = circshift(vec0, -row_idx)
                #@assert(vec1[end] < 10^-5) # zero entry should be at the end
                vec3 = circshift(reverse(vec1), n-1)
                #@assert(vec3[n] < 10^-5) # now the zero should be in the n'th entry
                UIJ[n, :] = vec3
            else
                for J in 1:dim
                    UIJ[n, J] = calculate_UIJ_element(n, J, wire_positions, num_wires, aₘ, lB, Lx, λ)
                end
            end
        end
    end

    # MAIN FUNCTION 

    # Populate first num_wires rows of matrix 
    fill_UIJ!(UIJ, num_wires, dim, wire_positions, aₘ, lB, Lx, λ, inversion_symmetric)

    # Populate the rest using the first rows
    for n in 1:num_wires
        for I in 2:N
            row_idx = num_wires * (I - 1) + n
            UIJ[row_idx, :] = circshift(UIJ[n, :], row_idx - n)
        end
    end

    if norm(UIJ-UIJ')<1e-7 # if it's basically symmetric
        UIJ = (UIJ+UIJ')/2; # make it exactly symmetric
    end
    @assert(issymmetric(UIJ))
    return UIJ
    
end

function make_V(wire_velocities::Vector{Float64},U::Matrix{Float64},g₀::Float64,U_max::Float64)::Matrix{Float64}
    U = g₀*U/U_max
    dim = size(U)[1]
    num_wires = length(wire_velocities)
    N = div(dim , num_wires)
    return kron(LinearAlgebra.I(N), Diagonal(abs.(wire_velocities))) + U/π 
end

function make_M(V::Matrix{Float64})::Matrix{Float64}
    # returns the scaling dimension matrix M corresponding to a positive-definite symmetric matrix V
    # See ``Almost Perfect Metals in One Dimension" by Murthy and Nayak (including Appendix S2A)
    # The only difference is that I assume K = diag(+1,-1,+1,-1,...) not diag(-I_n, I_n)

    tol = 1e-8
    @assert(issymmetric(V))
    @assert(isposdef(V))
    sqV = sqrt(V)
    K = make_K(size(V,1))

    @assert(issymmetric(sqV)) # check result is symmetric

    #@assert(norm(sqV*sqV-V) < tol) # and squares to V
    sqVinv = inv(sqV) # inverse
    L = sqVinv * K * sqVinv
    @assert(norm(L-transpose(L)) < tol)
    L = (L+transpose(L))/2 # force symmetrize
    
    Q, _, sqD = special_orthogonal_diagonalization(L) # In notation of App. S2A
    A = sqD * transpose(Q) * sqVinv

    #@assert(norm(det(A)-1)<tol) # Ensure det(A) = 1
    #@assert(norm(A*K*transpose(A) - K)<tol) # Ensure it preserves K
    #@assert(norm(A*V*transpose(A)-Diagonal(diag(A*V*transpose(A))))<tol) # And diagonalizes V
    M = transpose(A)*A

    return M
end

function special_orthogonal_diagonalization(L::Matrix{Float64})::Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64}}
    # Only works when M is 2n x 2n, symmetric, and has exactly n positive and n negative eigenvalues
    # Returns the Q and D^{-1} described in Appendix S2a of Murthy and Nayak 
    
    dim = size(L,1)
    n = div(dim,2)

    # Ensure L is symmetric
    @assert issymmetric(L)

    # Eigenvalue decomposition
    evals, evecs = eigen(L)
    # Note eigenvalues are in ascending order, so there are n negative ones followed by n positive ones

    # Reorder eigenvalues to alternate signs
    ordered_Dinv = zeros(Float64, dim,dim) # diagonal 
    ordered_sqDinv = zeros(Float64, dim,dim) # diagonal 
    ordered_Q = zeros(Float64, dim,dim)
    for i in 1:n
        i_p = 2*(i-1)+1
        i_n = 2*i
        ordered_Dinv[i_p,i_p] = 1/abs(evals[n+i])
        ordered_Dinv[i_n,i_n] = 1/abs(evals[i])
        ordered_sqDinv[i_p,i_p] = sqrt(1/abs(evals[n+i]))
        ordered_sqDinv[i_n,i_n] = sqrt(1/abs(evals[i]))      
        ordered_Q[:,i_p] = evecs[:,n+i]
        ordered_Q[:,i_n] = evecs[:,i]
    end

    Q = ordered_Q
    Dinv = ordered_Dinv
    sqDinv = ordered_sqDinv

    # Ensure Q is special orthogonal (Q*Q^T = I and det(Q) = 1)
    @assert isapprox(Q * transpose(Q), I, atol=1e-5)
    if det(Q) < 0
        Q[:, end] .= -Q[:, end]  # hit last column with a -1
    end

    return Q,  Dinv, sqDinv
end

function make_K(dim::Int)
    K = Diagonal([(-1)^(i+1) for i in 1:dim])
    return K
end

function all_possible_ops(wire_positions::Vector{Float64},d::Int)::Vector{Vector{Int}}
    # Return charge and dipole conserving vectors corresponding to 4-fermi operators given a list of wire positions within a unit cell,
    # `wire_positions`, and a cutoff `d` which is the range any operator (length of smallest interval containing its support)
    
    # Helper functions 
    function expand_list_to_length(list::Vector{Float64}, M::Int)::Vector{Float64}
        m0 = length(list)
        list2 = Vector{Float64}()
        for i in 0:div(M-1, m0)
            append!(list2, list .+ i)
        end
        return list2[1:M]
    end
    function calculate_inner_products(vectors::Vector{Vector{Int}}, list2::Vector{Float64})::Vector{Float64}
        inner_products = [dot(vec, list2) for vec in vectors]
        return inner_products
    end
    function generate_vectors(M::Int, nonzeros::Vector{Int}, M0::Int, m0::Int)
        vectors = Vector{Vector{Int}}()
    
        for i1 in 1:m0
            for i2 in i1+1:M-2
                for i3 in i2+1:M-1
                    i4 = i3 + (i2 - i1)
                    if i4 <= M && (i4 - i1) < M0
                        vec = zeros(Int, M)
                        vec[i1] = nonzeros[1]
                        vec[i2] = nonzeros[2]
                        vec[i3] = nonzeros[3]
                        vec[i4] = nonzeros[4]
                        push!(vectors, vec)
                    end
                end
            end
        end
        return vectors
    end

    # Parameters
    nonzeros = [1, -1, -1, 1]
    nw = length(wire_positions)
    M = d + 2*nw

    # Generate vectors
    result_vectors = generate_vectors(M, nonzeros, d, nw)

    # Expand list
    expanded_list = expand_list_to_length(wire_positions, M)

    inner_products = abs.(calculate_inner_products(result_vectors, expanded_list))

    symmetric_vectors = Vector{Vector{Int}}()
    for (ind,vec) in enumerate(result_vectors)
        if inner_products[ind] < 1e-8
            push!(symmetric_vectors,vec)
        end
    end
    return symmetric_vectors
end

function get_chern(m_op::Vector{Int})::Float64
    nonzero_indices = Int[]
    @inbounds for i in 1:length(m_op)
        if m_op[i] != 0 
            push!(nonzero_indices,i)
        end
    end
    l = nonzero_indices[2] - nonzero_indices[1]
    I0 = nonzero_indices[2]
    J0 = nonzero_indices[3]
    c = 0.0
    if mod(I0,2) == 0 # if I0 is even 
        if mod(l,2) == 0 # if l is even 
            c = (J0 - I0 + l + 1) / 2
        else 
            c = -(l-1)/2
        end
    else
        if mod(l,2) == 0 
            c = -(J0 - I0 + l - 1)/2
        else
            c = (l+1)/2
        end
    end
    return c
end

function get_chern_old(m_op::Vector{Int})::Float64
    # Compute chern number of a cosine operator packaged inside a vector m_op
    # We do this by counting gapless edge modes in a finite sample 

    v = m_op
    @assert(sum(v)==0) # check it conserves charge
    # Step 1: Pad v with zeros at the end 
    N = length(v)*40 # just need a large  N
    v_0 = vcat(v, zeros(Int, 2N - length(v)))

    # Step 2: Generate a collection of vectors, v_{i+1} is v_i shifted by 2 wires
    vectors = [v_0]
    for _ in 1:N-1
        v_next = circshift(vectors[end], 2)
        push!(vectors, v_next)
        if any(vectors[end][end-1:end] .!= 0) # Check the last two elements of the current vector
            break
        end
    end

    # Step 3: Add the vectors elementwise in absolute value to get V. 
    V = zeros(Int, 2N)
    for vec in vectors
        V .+= abs.(vec)
    end

    # Step 4: Count the number of zeros in V
    # if V has any zeros, those are gapless chiral modes, and that's how we'll compute Chern number 
    zero_count = count(x -> x == 0, V)

    # Step 5: Compute and return C
    C = zero_count / 2
    return C*(-1)^v[1] # accounting for sign of chern number 
    #= Example usage:
    v = [0,-1, 0,1,1,0, -1]
    C = get_chern(v)
    println("C = ", C)
    =#
end

function leading_instabilities(x,y::Vector{Float64}, nu::Float64, aₘ::Float64, lB::Float64, λ::Float64, inversion_symmetric::Bool, N::Int, g₀::Float64, U_max::Float64, d::Int, n::Int)::Tuple{Vector{Float64},Vector{Vector{Int64}},Vector{Float64}}
    #This function puts it all together. 
    # It takes in a dispersion of the form `x, y` (see find_wire_positions_and_velocities), filling `nu`, moire period `aₘ`, 
    # mag. length `lB`, scr.length `λ`, inversion symmetry indicator `inversion_symmetric`
    # hyperparameters: `N` ~ # of unit cells, `g₀` ∈(0,0.2ish] controls interaction strengths, `d` ~ range of instabilities, integer `n` 
    # Returns `n` leading instabilities, i.e. their scaling dimensions, operators, and chern numbers

    wire_positions, wire_velocities = find_wire_positions_and_velocities(x,y,nu)
    U = make_U(N,wire_positions,aₘ,lB,λ,inversion_symmetric) # density-density interactions
    V = make_V(wire_velocities,U,g₀,U_max) # V = |v|I + U, normalized
    M = make_M(V) # scaling dim matrix
    m_ops = all_possible_ops(wire_positions,d) # vectors representing instabilities
    sc_dims = Δ(m_ops,M) # scaling dimensions thereof

    sorted_indices = sortperm(sc_dims)[1:n] # indices of n leading instabilities 
    leading_dims = sc_dims[sorted_indices] # scaling dimensions,
    leading_ops = m_ops[sorted_indices] # vectors,  
    leading_cherns = Vector{Float64}() # and chern numbers of the n leading instabilities
    for i in 1:n
        push!(leading_cherns,get_chern(m_ops[sorted_indices[i]])) # and 
    end
    
    return leading_dims, leading_ops, leading_cherns
end

function U_of_x_y_quadgk(x::Float64, y::Float64; c::Int = 0)::Float64 # Screened Coulomb numerical integral
    integrand(s) = besselk(0, sqrt(((s + x)*y)^2 + eps())) * exp(-s^2 / 2)
    integral, _ = quadgk(integrand, -Inf, Inf,rtol=1e-5,atol=1e-8) # we ask that the error is 1 part in 10^5
    part2 = sqrt(π / 2) * exp(-(x^2 - y^2) / 4) * besselk(0, (x^2 + y^2) / 4) # see notes for formulas
    if c == 1 # direct piece only 
        return integral
    elseif c == 2 # exchange piece only
        return part2 
    end
    return max(integral - part2, 0) # Float errors sometimes give a tiny negative result when it should be = 0  
end

function F_of_x_y_z_quadgk(pl::Float64, ql::Float64, c::Float64)::Float64 # General Coulomb matrix element; numerical integral of A_1234
    # c = ℓ/λ
    integrand(s) = besselk(0, sqrt((pl + ql)^2 + 4*c^2 + eps()) * abs(s + (pl-ql)/2 + + eps()) + eps()) * exp(-s^2 / 2)
    integral, _ = quadgk(integrand, -Inf, Inf,rtol=0,atol=0) # we ask that the error is 1 part in 10^5
    total_val = exp(-(pl+ql)^2 / 8) * integral 
    return total_val
end

function fix_pixels(chern0::Matrix{Int64})::Matrix{Int64}
    # Get the dimensions of the matrix
    rows, cols = size(chern0)
    
    # Create a copy of the matrix to store the corrected values
    corrected_chern0 = copy(chern0)
    
    # Function to get the neighbors of an element
    function get_neighbors(matrix, i, j)
        neighbors = []
        for di in -1:1
            for dj in -1:1
                if di == 0 && dj == 0
                    continue
                end
                ni, nj = i + di, j + dj
                if ni > 0 && ni <= rows && nj > 0 && nj <= cols
                    push!(neighbors, matrix[ni, nj])
                end
            end
        end
        return neighbors
    end
    
    # Iterate through each element in the matrix
    for i in 1:rows
        for j in 1:cols
            neighbors = get_neighbors(chern0, i, j)
            value_counts = countmap(neighbors)
            
            # Check if more than 6 neighbors have the same value different from chern0[i,j]
            for (value, count) in value_counts
                if count > 6 && value != chern0[i, j]
                    corrected_chern0[i, j] = value
                    break
                end
            end
        end
    end
    return corrected_chern0
end

end


