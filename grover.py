from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram, plot_state_city, circuit_drawer
import matplotlib.pyplot as plt
import numpy as np


# ---------- Step 1: Create a general oracle for any target ----------
def make_oracle(target):
    n = len(target)
    qc = QuantumCircuit(n)
    # Flip qubits where target has '0'
    for i, bit in enumerate(target):
        if bit == "0":
            qc.x(i)

    # Mark the target state
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)

    # Undo the flips
    for i, bit in enumerate(target):
        if bit == "0":
            qc.x(i)
    return qc


# ---------- Step 2: Create the diffusion operator ----------
def diffusion_operator(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


# ---------- Step 3: Basic Grover search with histogram ----------
def grover_search(target):
    n = len(target)
    qc = QuantumCircuit(n, n)

    # Step 1: Put all qubits in superposition
    qc.h(range(n))

    # Step 2: Apply Oracle
    oracle = make_oracle(target)
    qc.compose(oracle, inplace=True)

    # Step 3: Apply Diffusion operator
    diffusion = diffusion_operator(n)
    qc.compose(diffusion, inplace=True)

    # Step 4: Measure
    qc.measure(range(n), range(n))

    # ---------- Run on simulator ----------
    backend = AerSimulator()
    qc_compiled = transpile(qc, backend)
    job = backend.run(qc_compiled, shots=1024)
    result = job.result()
    counts = result.get_counts()

    print("Counts:", counts)
    plot_histogram(counts)
    plt.title(f'Measurement Results (Target: {target})')
    plt.show()


# ---------- Visualization 1: Circuit Diagram ----------
def visualize_circuit(target):
    n = len(target)
    qc = QuantumCircuit(n, n)
    
    qc.h(range(n))
    oracle = make_oracle(target)
    qc.compose(oracle, inplace=True)
    diffusion = diffusion_operator(n)
    qc.compose(diffusion, inplace=True)
    qc.measure(range(n), range(n))
    
    fig = qc.draw('mpl', style='iq')
    plt.show()


# ---------- Visualization 2: Statevector Evolution ----------
def visualize_statevector(target):
    n = len(target)
    qc = QuantumCircuit(n)
    
    # Initial state
    qc.h(range(n))
    state_init = Statevector(qc)
    
    # After oracle
    oracle = make_oracle(target)
    qc.compose(oracle, inplace=True)
    state_after_oracle = Statevector(qc)
    
    # After diffusion
    diffusion = diffusion_operator(n)
    qc.compose(diffusion, inplace=True)
    state_final = Statevector(qc)
    
    # Plot all three states separately
    print("After Initialization:")
    fig1 = plot_state_city(state_init, title='After Initialization')
    plt.show()
    
    print("After Oracle:")
    fig2 = plot_state_city(state_after_oracle, title='After Oracle')
    plt.show()
    
    print("After Diffusion:")
    fig3 = plot_state_city(state_final, title='After Diffusion')
    plt.show()


# ---------- Visualization 3: Multiple Iterations Comparison ----------
def compare_iterations(target, max_iterations=5):
    n = len(target)
    results = {}
    
    for num_iter in range(1, max_iterations + 1):
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        
        # Apply Grover iteration multiple times
        for _ in range(num_iter):
            oracle = make_oracle(target)
            qc.compose(oracle, inplace=True)
            diffusion = diffusion_operator(n)
            qc.compose(diffusion, inplace=True)
        
        qc.measure(range(n), range(n))
        
        backend = AerSimulator()
        qc_compiled = transpile(qc, backend)
        job = backend.run(qc_compiled, shots=1024)
        counts = job.result().get_counts()
        
        # Store success probability
        success_rate = counts.get(target, 0) / 1024
        results[num_iter] = success_rate
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color='steelblue', edgecolor='black')
    plt.xlabel('Number of Iterations', fontsize=12)
    plt.ylabel('Success Probability', fontsize=12)
    plt.title(f'Grover Success Rate vs Iterations (target: {target})', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------- Visualization 4: Compare Different Target States ----------
def compare_targets(targets):
    fig, axes = plt.subplots(1, len(targets), figsize=(5*len(targets), 4))
    
    if len(targets) == 1:
        axes = [axes]
    
    for idx, target in enumerate(targets):
        n = len(target)
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        
        oracle = make_oracle(target)
        qc.compose(oracle, inplace=True)
        diffusion = diffusion_operator(n)
        qc.compose(diffusion, inplace=True)
        qc.measure(range(n), range(n))
        
        backend = AerSimulator()
        qc_compiled = transpile(qc, backend)
        job = backend.run(qc_compiled, shots=1024)
        counts = job.result().get_counts()
        
        # Plot histogram
        plot_histogram(counts, ax=axes[idx])
        axes[idx].set_title(f'Target: {target}')
    
    plt.tight_layout()
    plt.show()


# ---------- Visualization 5: Probability Evolution Over Time ----------
def probability_evolution(target):
    n = len(target)
    states = [format(i, f'0{n}b') for i in range(2**n)]
    
    # Track probabilities at each stage
    stages = ['Init', 'Oracle', 'Diffusion']
    probs = {state: [] for state in states}
    
    qc = QuantumCircuit(n)
    
    # After initialization
    qc.h(range(n))
    sv = Statevector(qc)
    for state in states:
        probs[state].append(abs(sv[int(state, 2)])**2)
    
    # After oracle
    oracle = make_oracle(target)
    qc.compose(oracle, inplace=True)
    sv = Statevector(qc)
    for state in states:
        probs[state].append(abs(sv[int(state, 2)])**2)
    
    # After diffusion
    diffusion = diffusion_operator(n)
    qc.compose(diffusion, inplace=True)
    sv = Statevector(qc)
    for state in states:
        probs[state].append(abs(sv[int(state, 2)])**2)
    
    # Plot
    plt.figure(figsize=(12, 6))
    x = range(len(stages))
    
    for state in states:
        plt.plot(x, probs[state], marker='o', label=state, 
                 linewidth=3 if state == target else 1,
                 alpha=1 if state == target else 0.5)
    
    plt.xlabel('Algorithm Stage', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Probability Evolution (Target: {target})', fontsize=14)
    plt.xticks(x, stages)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------- Visualization 6: Heatmap of All Possible Searches ----------
def success_heatmap(n=3):
    all_targets = [format(i, f'0{n}b') for i in range(2**n)]
    success_rates = []
    
    for target in all_targets:
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        oracle = make_oracle(target)
        qc.compose(oracle, inplace=True)
        diffusion = diffusion_operator(n)
        qc.compose(diffusion, inplace=True)
        qc.measure(range(n), range(n))
        
        backend = AerSimulator()
        qc_compiled = transpile(qc, backend)
        job = backend.run(qc_compiled, shots=1024)
        counts = job.result().get_counts()
        
        success_rates.append(counts.get(target, 0) / 1024)
    
    # Plot
    plt.figure(figsize=(10, 2))
    plt.imshow([success_rates], cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Success Rate')
    plt.xticks(range(len(all_targets)), all_targets)
    plt.yticks([])
    plt.xlabel('Target State', fontsize=12)
    plt.title('Grover Success Rate for All Possible Targets', fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------- MAIN: Run all visualizations ----------
if __name__ == "__main__":
    target = "101"
    
    print("=" * 60)
    print("GROVER'S ALGORITHM VISUALIZATIONS")
    print("=" * 60)
    print(f"\nTarget state: {target}\n")
    
    # 1. Basic histogram
    print("\n[1] Running basic Grover search...")
    grover_search(target)
    
    # 2. Circuit diagram
    print("\n[2] Displaying circuit diagram...")
    visualize_circuit(target)
    
    # 3. Statevector evolution
    print("\n[3] Visualizing statevector evolution...")
    visualize_statevector(target)
    
    # 4. Multiple iterations comparison
    print("\n[4] Comparing multiple iterations...")
    compare_iterations(target, max_iterations=5)
    
    # 5. Compare different targets
    print("\n[5] Comparing different target states...")
    compare_targets(['000', '010', '111'])
    
    # 6. Probability evolution
    print("\n[6] Showing probability evolution...")
    probability_evolution(target)
    
    # 7. Success heatmap
    print("\n[7] Generating success heatmap...")
    success_heatmap(n=3)
    
    print("\n" + "=" * 60)
    print("All visualizations complete!")
    print("=" * 60)