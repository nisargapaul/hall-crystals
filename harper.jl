module harper
using LinearAlgebra, SIMD
export solve_harpers_equation, compute_bloch_functions, calculate_charge_density, calculate_partial_charge_density, charge_average

# Function to solve Harper's equation
function solve_harpers_equation(U1::Float64, U2::Float64, a::Float64, b::Float64, q::Int, p::Int, k1::AbstractVector{Float64}, k2::AbstractVector{Float64})::Tuple{Array{Float64, 3}, Array{Complex{Float64}, 4}}
    eigenvalues = Array{Float64}(undef, length(k1), length(k2), p)
    eigenvectors = Array{Complex{Float64}, 4}(undef, length(k1), length(k2), p, p)
    
    @inbounds for (i, k1_val) in enumerate(k1)
        for (j, k2_val) in enumerate(k2)
            H = zeros(ComplexF64, p, p)
            @inbounds @simd for n in 1:p
                H[n, n] = 2 * U1 * cos(q * b * k2_val / p + 2π * n * q / p)
                if n > 1
                    H[n-1, n] = U2 * exp(-im * q * a * k1_val / p)
                end
                if n < p
                    H[n+1, n] = U2 * exp(im * q * a * k1_val / p)
                end
            end
            H[1, p] += U2 * exp(im * q * a * k1_val / p)
            H[p, 1] += U2 * exp(-im * q * a * k1_val / p) 
            # Ensure Hermiticity
            if !isapprox(H, H')
                error("Matrix H is not Hermitian for k1 = $k1_val, k2 = $k2_val.")
            end
            # Compute eigenvalues and eigenvectors
            evals, evecs = eigen(H)
            evals = real(evals)
            idx = sortperm(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]
            
            eigenvalues[i, j, :] = real(evals)
            eigenvectors[i, j, :, :] = evecs
        end
    end
    
    return eigenvalues, eigenvectors
end


# Function to compute Bloch functions
function compute_bloch_functions(eigenvectors::Array{Complex{Float64}, 4}, x::AbstractVector{Float64}, y::AbstractVector{Float64}, k1::AbstractVector{Float64}, k2::AbstractVector{Float64}, a::Float64, b::Float64, q::Int, p::Int, Mx::Int)::Array{Complex{Float64}, 5}
    Nk1 = length(k1)
    Nk2 = length(k2)
    Nx = length(x)
    Ny = length(y)
    bloch_functions = Array{Complex{Float64}, 5}(undef, Nk1, Nk2, p, Nx, Ny)
    B = 2*pi*p/(q*a*b) # magnetic field
    sum_terms = 0.0 + 0.0im
    argx = 0.0 
    exp_term = 0.0 + 0.0im
    exp_term2_base = 0.0 + 0.0im 
    exp_term2 = 0.0 + 0.0im 
    chi_term = 0.0 

    @inbounds for (i, k1_val) in enumerate(k1)
        @inbounds for (j, k2_val) in enumerate(k2)
            @inbounds for w in 1:p
                d_vec = @view eigenvectors[i, j, :, w]
                @inbounds for (ix, x_val) in enumerate(x)
                    @inbounds for (iy, y_val) in enumerate(y)
                        sum_terms = 0.0 + 0.0im
                        exp_term2_base = exp(2π * im * (y_val / b))
                        @inbounds @simd for n in 1:p
                            for l in -(2*Mx):(3*Mx)  # Sum over many periods
                                argx = x_val - k2_val / B - l * q * a - n * q * a / p
                                exp_term = exp(-im * k1_val * (x_val - l * q * a - n * q * a / p))
                                exp_term2 = exp_term2_base^(l * p + n)
                                chi_term = exp(- B * argx^2 / 2)
                                sum_terms += d_vec[n] * chi_term * exp_term * exp_term2
                            end
                        end
                        bloch_functions[i, j, w, ix, iy] = sum_terms
                    end
                end
            end
        end
    end
    
    return bloch_functions
end


function calculate_charge_density(bloch_functions::Array{Complex{Float64}, 5}, k1::AbstractVector, k2::AbstractVector, r::Int, p::Int)::Array{Float64, 2}
    Nk1, Nk2, _, Nx, Ny = size(bloch_functions)
    charge_density = zeros(Float64, Nx, Ny)
    
    for (i, _) in enumerate(k1)
        for (j, _) in enumerate(k2)
            for w in 1:r
                for ix in 1:Nx
                    for iy in 1:Ny
                        charge_density[ix, iy] += abs2(bloch_functions[i, j, w, ix, iy])
                    end
                end
            end
        end
    end
    
    # Normalize the charge density
    charge_density /= (Nk1 * Nk2)
    
    return charge_density
end

function calculate_partial_charge_density(
    bloch_functions::Array{Complex{Float64}, 5}, 
    eigenvalues::Array{Float64, 3}, 
    ν::Float64)::Array{Float64, 2}
    # Check if ν is within valid range
    if ν < 0.0 || ν > 1.0
        error("Filling factor ν must be between 0 and 1.")
    end

    Nk1, Nk2, p, Nx, Ny = size(bloch_functions)
    total_states = Nk1 * Nk2 * p
    num_occupied_states = round(Int, total_states * ν)
    
    # Initialize the charge density array
    charge_density = zeros(Float64, Nx, Ny)

    # Create a list of (k1, k2, n) tuples corresponding to state indices
    state_indices = [(i, j, w) for i in 1:Nk1 for j in 1:Nk2 for w in 1:p]

    # Flatten the eigenvalues and associate them with state indices
    flattened_eigenvalues = [eigenvalues[i, j, w] for i in 1:Nk1 for j in 1:Nk2 for w in 1:p]
    
    # Sort the flattened eigenvalues and get the indices of the lowest energy states
    sorted_indices = sortperm(flattened_eigenvalues)

    # Take the indices corresponding to the lowest num_occupied_states energies
    occupied_states = state_indices[sorted_indices[1:num_occupied_states]]

    # Sum the contributions from the occupied states
    for (i, j, w) in occupied_states
        for ix in 1:Nx
            for iy in 1:Ny
                charge_density[ix, iy] += abs2(bloch_functions[i, j, w, ix, iy])
            end
        end
    end

    # Normalize the charge density
    charge_density /= (Nk1 * Nk2)

    return charge_density
end


function charge_average(charge_density::Array{Float64, 2}, q::Int, Mx::Int)::Array{Float64, 2}
    Nx, Ny = size(charge_density)
    averaged_charge_density = zeros(Float64, Nx, Ny)

    for shift in 0:(q-1)
        # Calculate the shift in index terms
        shift_amount = round(Int, shift * Nx/Mx)

        for ix in 1:Nx
            translated_ix = mod(ix + shift_amount - 1, Nx) + 1
            averaged_charge_density[translated_ix, :] += charge_density[ix, :]
        end
    end
    
    # Average over the q translated versions
    averaged_charge_density /= q

    return averaged_charge_density
end



end 
