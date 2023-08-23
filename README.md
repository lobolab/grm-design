# Automatic design of GRMs for spatial patterns

Supplementary Data for:

> Automatic design of gene regulatory mechanisms for spatial pattern formation<br>
> Reza Mousavi and Daniel Lobo<br>
[Lobo Lab](https://lobolab.umbc.edu)<br>
> *Under review*

We present a general framework to automatically design GRMs able to produce any given spatial pattern. The automated method is based on high-performance evolutionary computation to rapidly design GRMs--including the necessary number of genes and interaction mechanisms.

## Building
Two different solutions are included:
* Evolution: an evolutionary algorithm for designing GRMs
* Viewer: user interface for visualizing the evolutionary algorithm results

Open each solution and compile them with Microsoft Visual Studio. Make sure the required dependencies are installed on your computer.

## Dependencies
* Microsoft Visual Studio
* NVIDIA CUDA Toolkit
* Eigen
* Qt
* Qwt

## Implementation
The `Src` folder contains all the C++ source code. It is further organized into the following folders:

### `Common` 
This folder includes helper classes to handle files, save logs, and perform general mathematical methods.

### `DB` 
This folder contains the code for accessing the database files that store the settings and results of the automated algorithm. These include the parameters needed to run the method, the GRMs produced during the evolutionary search, and the evolutionary statistics computed during execution. The viewer can open these files to display the results of an evolutionary run.

### `Experiment` 
This folder includes the code to specify the target spatial gene expression pattern defined in an experiment. The input target pattern is implemented as an image, in which each color corresponds to a different gene.

### `GPUSimulator` 
This folder contains the simulator that runs on the GPU. This includes the simulation of GRMs and the computation of their fitness (error with respect the target gene expression pattern). In addition, this folder includes classes for the initialization of GPUs, the copying of input gene expression patterns, and the transfer of GRMs and their numerical fitness between the CPU and GPU memory.

### `Model` 
This folder includes the implementation of GRMs. A mechanism includes a set of genes and their parameters and a set of links specifying the type of regulatory interactions between two genes and their parameters.

### `Search` 
This folder contains the code for loading the parameters of the evolutionary algorithm, handling the mechanism populations on the different islands, generating new mechanisms by executing evolutionary operators (crossover and mutation), and selecting the next generation populations. The folder also includes an implementation of the mechanism fitness calculator that runs on the CPU.

### `Simulator` 
This folder includes the simulator that runs on the CPU. This includes the implementation for generating the two-dimensional orthogonal input morphogen gradients, loading parameters related to the simulation, loading the GRMs defined as classes into a system of PDEs for simulation, and performing the numerical simulation using an Euler finite difference method.

### `UI` 
The UI folder contains the user interface for both the evolution and viewer. The evolution program is run with a command line interface that uses a multi-thread implementation for maximizing performance. The viewer includes a graphical user interface to visualize the results of the evolution and perform simulations of the discovered mechanisms.