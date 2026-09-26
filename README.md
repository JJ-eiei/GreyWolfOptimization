# Grey Wolf Optimization for Numerical Benchmark Functions

A from-scratch Python implementation of **Grey Wolf Optimization (GWO)**, tested on the Rosenbrock and Rastrigin benchmark functions.

This project demonstrates how a population-based optimization algorithm can search for high-quality solutions without using gradients. It also includes repeatable experiments, statistical evaluation, and visualizations that make the algorithm's behavior easier to understand.

## Project Overview

Grey Wolf Optimization is a nature-inspired metaheuristic based on the leadership hierarchy and hunting behavior of grey wolves. In this project, a population of candidate solutions (the “wolves”) moves through a two-dimensional search space. The three strongest candidates—Alpha, Beta, and Delta—guide the rest of the population toward promising regions.

The implementation evaluates GWO on two well-known optimization problems:

- **Rosenbrock:** a smooth function with a narrow, curved valley that can be difficult to navigate.
- **Rastrigin:** a highly repetitive, multi-modal function with many local minima.

Together, they test the algorithm under two different search challenges: precise movement through a difficult valley and avoiding attractive but incorrect local solutions.

## Objectives

- Implement Grey Wolf Optimization directly with NumPy.
- Show how Alpha, Beta, and Delta wolves influence population movement.
- Test the algorithm on benchmark functions with known global optima.
- Visualize convergence, trajectories, and distance from the optimum.
- Evaluate reliability across 30 independent runs rather than relying on one result.
- Produce a step-by-step manual calculation for learning and verification.

## Implemented Features

- Random population initialization within defined search bounds
- Fitness-based selection of Alpha, Beta, and Delta wolves
- Standard GWO position-update equations
- Linear reduction of the exploration parameter `a`
- Boundary handling with position clipping
- Tracking of the best solution found across the complete run
- Final-population evaluation after the last position update
- Position, leader, fitness, convergence, and distance histories
- One movement plot for the initial population and every iteration
- Wolf trajectories over objective-function contour maps
- Alpha, Beta, Delta, and known optimum markers
- Animated GIF generation from saved movement frames
- Distance-to-optimum plots and tab-separated data exports
- Average convergence comparison across 30 runs
- Reproducible random seeds
- Two-iteration manual calculation reports for both benchmark functions

## Experiment Setup

| Setting | Value |
|---|---:|
| Dimensions | 2 |
| Wolves per run | 30 |
| Maximum iterations | 50 |
| Independent evaluation runs | 30 |
| Lower bound | -5.0 |
| Upper bound | 5.0 |
| Single-run seed | 42 |
| Evaluation seeds | 1000–1029 |
| Animation speed | 4 frames per second |

Each illustrated single run starts with 30 randomly positioned wolves. The statistical evaluation repeats the full optimization 30 times using a different deterministic seed for each run.

## Benchmark Functions

### Rosenbrock Function

```text
f(x, y) = (1 - x)² + 100(y - x²)²
```

- Known global optimum: `(1, 1)`
- Minimum fitness: `0`
- Main challenge: following a narrow, curved valley to reach the optimum

### Rastrigin Function

```text
f(x, y) = 20 + x² + y² - 10[cos(2πx) + cos(2πy)]
```

- Known global optimum: `(0, 0)`
- Minimum fitness: `0`
- Main challenge: avoiding many local minima created by the oscillating landscape

For both functions, a lower fitness value is better.

## How the Algorithm Works

1. Initialize the wolves at random positions in the search space.
2. Evaluate every wolf using the selected benchmark function.
3. Rank the population and assign the three best wolves as Alpha, Beta, and Delta.
4. Reduce the control parameter `a` linearly from `2` toward `0`:

   ```text
   a = 2 - 2t/T
   ```

5. For each wolf and each leader, generate coefficient vectors:

   ```text
   A = 2ar₁ - a
   C = 2r₂
   D = |C · X_leader - X|
   X_candidate = X_leader - A · D
   ```

6. Update the wolf using the average position suggested by the three leaders:

   ```text
   X(t + 1) = (X₁ + X₂ + X₃) / 3
   ```

7. Clip updated positions to the allowed search range.
8. Repeat the process for 50 iterations and evaluate the final population once more.

At the beginning of a run, larger values of `a` support broader exploration. As `a` decreases, the population increasingly focuses on areas identified by its leaders.

## Outputs and Visualizations

Running the program creates the `GWO_results2/` directory with:

- **Manual calculation reports:** detailed calculations for the first two iterations, using five fixed starting wolves and calculator-friendly random values rounded to two decimal places.
- **Movement frames:** 51 PNG images per benchmark—the initial population plus the population after each of 50 updates.
- **Trajectory plots:** each movement frame includes the paths taken by individual wolves up to that point.
- **Leader markers:** Alpha, Beta, and Delta are highlighted separately from the remaining population.
- **Global optimum marker:** the known target location is shown on each contour plot.
- **GIF animations:** the saved frames are combined into one animation per benchmark.
- **Distance plots:** average population distance and closest-wolf distance from the known optimum.
- **Distance data files:** iteration, average distance, closest-wolf distance, and current best fitness in tab-separated format.
- **Average convergence plot:** a logarithmic-scale comparison of mean global-best fitness across the 30 runs.

Log scaling is used only to improve visualization; it does not change the fitness values used by GWO.

## 30-Run Evaluation Metrics

The program calculates the following metrics independently for Rosenbrock and Rastrigin:

| Metric | Meaning |
|---|---|
| Best | Lowest final global-best fitness across all runs |
| Mean | Average final global-best fitness |
| Worst | Highest final global-best fitness |
| Standard deviation | Variation in final fitness across runs |
| Average runtime | Mean execution time of one optimization run |
| Average convergence | Mean global-best fitness at each recorded iteration |

The numerical results are generated and printed when the program runs. They are intentionally not hard-coded in this README because runtime and optimization outcomes should come from the actual execution environment.

## Tech Stack

- **Python 3** — implementation and experiment workflow
- **NumPy** — numerical operations, random-number generation, and population updates
- **Matplotlib** — contour maps, movement frames, convergence plots, and distance plots
- **Pillow** — GIF creation from generated PNG frames
- **Python standard library** — file management and runtime measurement

## Project Structure

```text
.
├── README.md
├── <main_script>.py
└── GWO_results2/                     # Created when the script runs
    ├── Manual_Rosenbrock.md
    ├── Manual_Rastrigin.md
    ├── Rosenbrock_Every_Iteration/
    │   ├── frame_00.png
    │   ├── ...
    │   └── frame_50.png
    ├── Rastrigin_Every_Iteration/
    │   ├── frame_00.png
    │   ├── ...
    │   └── frame_50.png
    ├── Rosenbrock_GWO.gif
    ├── Rastrigin_GWO.gif
    ├── Rosenbrock_Distance.png
    ├── Rastrigin_Distance.png
    ├── Rosenbrock_Distance_Data.txt
    ├── Rastrigin_Distance_Data.txt
    └── Average_Convergence_30_Runs.png
```

Replace `<main_script>.py` with the filename used for the supplied Python source.

## How to Run

1. Clone or download the repository.
2. Open a terminal in the project directory.
3. Optionally create and activate a virtual environment.
4. Install the dependencies:

   ```bash
   pip install numpy matplotlib pillow
   ```

5. Run the Python script:

   ```bash
   python <main_script>.py
   ```

6. Review the console summary and the generated files inside `GWO_results2/`.

## Dependencies

```text
numpy
matplotlib
pillow
```

Pillow is loaded only when GIF creation is requested. If it is unavailable, the optimization and other outputs can still run, while the program displays an installation warning and skips GIF generation.

## What This Project Demonstrates

- Understanding of population-based and nature-inspired optimization
- Translation of mathematical equations into a working algorithm
- Reproducible experiment design through controlled random seeds
- Fairer performance assessment through repeated independent runs
- Statistical reporting of solution quality and consistency
- Visual communication of algorithm behavior and convergence
- Structured data collection for later analysis
- Attention to implementation details such as search boundaries and final-state evaluation

## Possible Future Improvements

- Support benchmark problems with more than two dimensions while retaining 2D visualization where applicable.
- Add command-line options for population size, iteration count, bounds, function, and seed.
- Save evaluation metrics to CSV or JSON in addition to printing them.
- Compare GWO with algorithms such as Particle Swarm Optimization or Genetic Algorithms under the same settings.
- Add automated tests for objective functions, boundary handling, history lengths, and reproducibility.
- Track additional measures such as success rate and distance of the best solution from the known optimum.
- Separate optimization, visualization, and experiment code into reusable modules.
- Add type hints and a dependency file for easier maintenance and setup.

## Note

GWO is a stochastic optimization method, so different random seeds can produce different solutions. The fixed seeds in this project make the included experiment workflow reproducible while the 30-run evaluation provides a more meaningful view of performance than a single run alone.

