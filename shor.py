"""
Valid Shor's Algorithm Simulation
Compares theoretical complexity and operation counts
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
import math

@dataclass
class AlgorithmMetrics:
    """Store algorithm performance metrics"""
    N: int
    method: str
    operations: int  # Actual operations performed
    theoretical_ops: float  # Theoretical complexity
    success_probability: float
    description: str

def classical_period_finding_ops(a: int, N: int) -> Tuple[int, int]:
    """
    Classical period finding - count operations.
    Returns: (period, operation_count)
    """
    ops = 0
    current = a % N
    ops += 2  # modulo and assignment
    
    for r in range(1, N):  # Worst case: try up to N
        ops += 2  # comparison and increment
        if current == 1:
            return r, ops
        current = (current * a) % N
        ops += 3  # multiply, modulo, assignment
    
    return None, ops

def quantum_period_finding_ops(N: int, n_count: int = None) -> Tuple[int, float]:
    """
    Quantum period finding - count quantum operations.
    Returns: (operation_count, theoretical_complexity)
    """
    if n_count is None:
        n_count = 2 * math.ceil(math.log2(N))
    
    n_work = math.ceil(math.log2(N))
    
    # Count quantum gates
    ops = 0
    
    # Hadamard gates on counting qubits
    ops += n_count
    
    # Controlled modular exponentiation (most expensive)
    # Each requires O(log³ N) gates in practice
    for i in range(n_count):
        ops += (math.log2(N) ** 3)
    
    # Inverse QFT: O(n_count²) gates
    ops += n_count ** 2
    
    # Measurements
    ops += n_count
    
    # Theoretical complexity: O(log³ N)
    theoretical = (math.log2(N) ** 3)
    
    return int(ops), theoretical

def compute_theoretical_complexity(N_values):
    """Compute theoretical complexities for different algorithms"""
    
    results = []
    
    for N in N_values:
        log_N = math.log2(N)
        
        # Classical period finding: O(N) worst case
        classical_ops = N
        classical_theory = N
        
        # Quantum period finding: O(log³ N) operations
        # But needs to be run O(log N) times for success
        quantum_ops_single = (log_N ** 3)
        quantum_ops_total = quantum_ops_single * log_N  # O(log⁴ N)
        quantum_theory = log_N ** 3
        
        # Classical factoring (GNFS): O(exp(∛(log N · log log N)))
        gnfs_theory = math.exp((64/9 * log_N * math.log(log_N)) ** (1/3))
        
        results.append({
            'N': N,
            'classical_period': classical_theory,
            'quantum_period': quantum_theory,
            'gnfs': gnfs_theory,
            'log_N': log_N
        })
    
    return results

def plot_additional_analysis():
    """Additional useful visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Shor\'s Algorithm: Extended Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Graph 1: Success Probability vs Precision (n_count qubits)
    ax1 = axes[0, 0]
    n_count_values = range(4, 20)
    # Success probability increases with more counting qubits
    success_probs = [1 - 1/(2**(n-1)) for n in n_count_values]
    
    ax1.plot(n_count_values, success_probs, 'o-', linewidth=2.5, 
             markersize=8, color='#A23B72')
    ax1.fill_between(n_count_values, success_probs, alpha=0.3, color='#A23B72')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax1.set_xlabel('Counting Qubits (n)', fontweight='bold')
    ax1.set_ylabel('Success Probability', fontweight='bold')
    ax1.set_title('Success Rate vs Counting Qubits', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # Graph 2: Resource Requirements Growth
    ax2 = axes[0, 1]
    bit_sizes = np.arange(8, 128, 4)
    qubits_needed = [2 * bits for bits in bit_sizes]  # Simplified: 2n qubits
    gates_needed = [bits**3 for bits in bit_sizes]  # O(n³) gates
    
    # ax2_twin = ax2.twinx()
    p1 = ax2.plot(bit_sizes, qubits_needed, 'o-', label='Qubits', 
                  linewidth=2, markersize=6, color='#2E86AB')
    # p2 = ax2_twin.semilogy(bit_sizes, gates_needed, 's-', label='Gates (log scale)', 
    #                        linewidth=2, markersize=6, color='#F18F01')
    
    ax2.set_xlabel('Input Size (bits)', fontweight='bold')
    ax2.set_ylabel('Qubits Required', fontweight='bold', color='#2E86AB')
    # ax2_twin.set_ylabel('Gate Count (log)', fontweight='bold', color='#F18F01')
    ax2.set_title('Quantum Resource Requirements', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2E86AB')
    # ax2_twin.tick_params(axis='y', labelcolor='#F18F01')
    ax2.grid(True, alpha=0.3)
    
    lines = p1
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    # Graph 3: Time to Break RSA (Hypothetical Real Quantum Computer)
    ax3 = axes[0, 2]
    rsa_sizes = [512, 1024, 2048, 3072, 4096]
    
    # Classical GNFS time (very rough estimates in years)
    classical_years = [0.01, 10, 300_000, 10**9, 10**12]
    
    # Quantum time (hypothetical, in hours)
    quantum_hours = [0.1, 0.5, 2, 8, 24]
    
    x = np.arange(len(rsa_sizes))
    width = 0.35
    
    ax3.bar(x - width/2, np.log10(classical_years), width, 
            label='Classical (GNFS)', color='#2E86AB', alpha=0.8)
    ax3.bar(x + width/2, np.log10(quantum_hours) - 8.76, width, 
            label='Quantum (hypothetical)', color='#A23B72', alpha=0.8)
    
    ax3.set_xlabel('RSA Key Size (bits)', fontweight='bold')
    ax3.set_ylabel('Time (log₁₀ years)', fontweight='bold')
    ax3.set_title('Time to Factor RSA Keys', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(rsa_sizes)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Graph 4: Error Rate Impact
    ax4 = axes[1, 0]
    error_rates = np.logspace(-4, -1, 20)  # 0.01% to 10%
    # Simplified: success = (1-error)^(num_gates)
    num_gates_scenarios = [100, 1000, 10000]
    
    for gates in num_gates_scenarios:
        success_with_error = [(1 - err)**gates for err in error_rates]
        ax4.semilogx(error_rates * 100, success_with_error, 'o-', 
                    label=f'{gates} gates', linewidth=2, markersize=5)
    
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Gate Error Rate (%)', fontweight='bold')
    ax4.set_ylabel('Circuit Success Probability', fontweight='bold')
    ax4.set_title('Impact of Quantum Errors', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)
    
    # Graph 5: Quantum Advantage Boundary
    ax5 = axes[1, 1]
    N_range = np.logspace(1, 12, 100)
    
    # Classical: O(N) for period finding
    classical_ops = N_range
    
    # Quantum: O(log³ N)
    quantum_ops = np.log2(N_range)**3
    
    # Fill regions
    ax5.fill_between(N_range, 0, quantum_ops, alpha=0.3, 
                     color='#A23B72', label='Quantum domain')
    ax5.fill_between(N_range, quantum_ops, classical_ops, 
                     alpha=0.3, color='#2E86AB', label='Classical faster')
    
    ax5.loglog(N_range, classical_ops, linewidth=3, color='#2E86AB', 
              label='Classical O(N)')
    ax5.loglog(N_range, quantum_ops, linewidth=3, color='#A23B72', 
              label='Quantum O(log³ N)')
    
    # Mark crossover
    crossover = next(i for i, (c, q) in enumerate(zip(classical_ops, quantum_ops)) if q < c)
    ax5.plot(N_range[crossover], quantum_ops[crossover], 'r*', 
            markersize=20, label=f'Crossover ≈ {N_range[crossover]:.0f}')
    
    ax5.set_xlabel('Problem Size (N)', fontweight='bold')
    ax5.set_ylabel('Operations Required', fontweight='bold')
    ax5.set_title('Quantum Advantage Boundary', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, which='both')
    
    # Graph 6: Measurement Outcome Distribution
    ax6 = axes[1, 2]
    
    # Simulate measurement distribution for period finding
    # For N=15, a=7, period r=4
    n_count = 8
    r_actual = 4
    phases = np.arange(0, 2**n_count) / (2**n_count)
    
    # Probability peaks at multiples of 1/r
    probabilities = np.zeros(2**n_count)
    for k in range(r_actual):
        # Peak at phase = k/r
        peak_phase = k / r_actual
        peak_index = int(peak_phase * (2**n_count))
        # Add a narrow peak
        for i in range(-2, 3):
            idx = (peak_index + i) % (2**n_count)
            probabilities[idx] += np.exp(-(i**2) / 0.5)
    
    probabilities = probabilities / np.sum(probabilities)
    
    ax6.stem(phases[:64], probabilities[:64], linefmt='#A23B72', 
            markerfmt='o', basefmt=' ')
    
    # Mark the peaks corresponding to period
    for k in range(r_actual):
        peak_phase = k / r_actual
        if peak_phase < 0.25:  # Only show first quarter
            ax6.axvline(x=peak_phase, color='red', linestyle='--', 
                       alpha=0.5, linewidth=2)
            ax6.text(peak_phase, max(probabilities) * 1.1, f'{k}/{r_actual}', 
                    ha='center', fontsize=9, color='red', fontweight='bold')
    
    ax6.set_xlabel('Measured Phase', fontweight='bold')
    ax6.set_ylabel('Probability', fontweight='bold')
    ax6.set_title('QFT Measurement Distribution (N=15, r=4)', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_xlim(0, 0.25)
    
    plt.tight_layout()
    plt.show()

def plot_theoretical_comparison():
    """Plot theoretical complexity comparison"""
    
    # Use a range that shows the crossover point
    N_values = [2**i for i in range(4, 41, 2)]  # 16 to ~1 trillion
    plot3_N_values=[2**i for i in range(4, 101, 4)]
    
    results = compute_theoretical_complexity(N_values)
    plot3_results=compute_theoretical_complexity(plot3_N_values)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Shor\'s Algorithm: Theoretical Complexity Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Extract data
    Ns = [r['N'] for r in results]
    classical = [r['classical_period'] for r in results]
    classical_plot3=[r['classical_period'] for r in plot3_results]
    quantum = [r['quantum_period'] for r in results]
    quantum_plot3 = [r['quantum_period'] for r in plot3_results]
    gnfs = [r['gnfs'] for r in results]
    gnfs_plot3 = [r['gnfs'] for r in plot3_results]
    log_N = [r['log_N'] for r in results]
    log_N_plot3=[r['log_N'] for r in plot3_results]
    
    # Plot 1: Period Finding Comparison (Log Scale)
    ax1 = axes[0, 0]
    ax1.loglog(Ns, classical, 'o-', label='Classical O(N)', 
               linewidth=2, markersize=6, color='#2E86AB')
    ax1.loglog(Ns, quantum, 's-', label='Quantum O(log³ N)', 
               linewidth=2, markersize=6, color='#A23B72')
    ax1.set_xlabel('Number to Factor (N)', fontweight='bold')
    ax1.set_ylabel('Operations (log scale)', fontweight='bold')
    ax1.set_title('Period Finding: Quantum vs Classical', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # Add crossover annotation
    crossover_idx = next((i for i, (c, q) in enumerate(zip(classical, quantum)) 
                         if q < c), None)
    if crossover_idx:
        ax1.annotate('Quantum advantage begins',
                    xy=(Ns[crossover_idx], quantum[crossover_idx]),
                    xytext=(Ns[crossover_idx]*10, quantum[crossover_idx]*10),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, color='red', fontweight='bold')
    
    # Plot 2: Speedup Factor
    ax2 = axes[0, 1]
    speedup = [c/q for c, q in zip(classical, quantum)]
    ax2.semilogy(log_N, speedup, 'o-', linewidth=2, markersize=6, color='#F18F01')
    ax2.axhline(y=1, color='red', linestyle='--', label='No speedup')
    ax2.set_xlabel('log₂(N) - Bit Size', fontweight='bold')
    ax2.set_ylabel('Speedup Factor (log scale)', fontweight='bold')
    ax2.set_title('Quantum Speedup Over Classical Period Finding', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(log_N, 1, speedup, where=[s > 1 for s in speedup], 
                     alpha=0.3, color='#F18F01', label='Quantum advantage')
    
    # Plot 3: Full Factoring Comparison
    ax3 = axes[1, 0]
    ax3.semilogy(log_N_plot3, classical_plot3, 'o-', label='Classical Period O(N)', 
                linewidth=2, markersize=6, color='#2E86AB')
    ax3.semilogy(log_N_plot3, quantum_plot3, 's-', label='Quantum Period O(log³ N)', 
                linewidth=2, markersize=6, color='#A23B72')
    ax3.semilogy(log_N_plot3, gnfs_plot3, '^-', label='GNFS (Best Classical)', 
                linewidth=2, markersize=6, color='#C73E1D')
    ax3.set_xlabel('log₂(N) - Bit Size', fontweight='bold')
    ax3.set_ylabel('Operations (log scale)', fontweight='bold')
    ax3.set_title('Factoring Complexity: Including GNFS', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Real-World Impact Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create table data
    bit_sizes = [512, 1024, 2048, 4096]
    table_data = []
    
    for bits in bit_sizes:
        N = 2 ** bits
        classical_ops = N
        quantum_ops = (bits ** 3)
        gnfs_ops = math.exp((64/9 * bits * math.log(bits)) ** (1/3))
        
        table_data.append([
            f"{bits} bits",
            f"2^{bits}",
            f"{bits}³ ≈ {quantum_ops:.0e}",
            f"{gnfs_ops:.2e}"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Key Size', 'Classical\nPeriod (ops)', 
                               'Quantum\nPeriod (ops)', 'GNFS (ops)'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.3, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows
    colors = ['#f0f0f0', 'white']
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            table[(i, j)].set_facecolor(colors[i % 2])
    
    ax4.set_title('Real-World Cryptographic Key Sizes', 
                  fontweight='bold', pad=20, fontsize=12)
    
    plt.tight_layout()
    plt.show()

def demonstrate_operation_counting():
    """Demonstrate actual operation counting on small numbers"""
    
    print("\n" + "="*80)
    print("OPERATION COUNTING DEMONSTRATION")
    print("="*80)
    print("\nComparing actual operations performed (not wall-clock time)\n")
    
    test_cases = [(7, 15), (8, 21), (5, 77)]
    
    results = []
    
    for a, N in test_cases:
        print(f"Finding period of a={a} mod N={N}")
        print("-" * 60)
        
        # Classical
        r_classical, classical_ops = classical_period_finding_ops(a, N)
        print(f"  Classical: r={r_classical}, operations={classical_ops}")
        
        # Quantum (theoretical)
        quantum_ops, quantum_theory = quantum_period_finding_ops(N)
        print(f"  Quantum: operations≈{quantum_ops}, theoretical=O(log³({N}))≈{quantum_theory:.1f}")
        
        ratio = classical_ops / quantum_ops if quantum_ops > 0 else 0
        print(f"  Speedup: {ratio:.2f}x\n")
        
        results.append({
            'N': N,
            'classical_ops': classical_ops,
            'quantum_ops': quantum_ops,
            'speedup': ratio
        })
    
    print("="*80)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("="*80)
    
    # Show operation counting
    demonstrate_operation_counting()
    
    # Show theoretical comparison
    print("\nGenerating theoretical complexity plots...")
    plot_theoretical_comparison()
    
    # Show additional analysis
    print("\nGenerating extended analysis plots...")
    plot_additional_analysis()
