> THIS IS A MINI SUMMARY / ABSTRACT OF THE PROJECT AND DOES NOT CONTAIN ALL INFORMATION OR EXAMPLES, STRICTLY USE THIS TO UNDERSTAND THE OVERVIEW FOR THE OUTPUT REQUIRED AND THE FRONTEND EXPECTED MORE THAN THE METHODOLOGY OR THE PREPROCESSING OF THE DATASETS OR THE FEATURES EXTRACTED. YOU WILL HAVE TO EXTRACT THOSE 

# Title
A Data-Driven Comparative Analysis of Hybrid Continual Learning Methods: Mitigating Catastrophic Forgetting

# Compute-Bound Dataset Rationale
The dataset mix is intentionally driven by limited hardware budgets. Smaller datasets already run locally, and Mini-ImageNet ResNet18 is now also intended to run locally on the RTX 4050 laptop with resumable checkpoints. Heavier ViT/Tiny-ImageNet work can still use a fallback cloud path if local memory becomes a bottleneck.

# About the Data Set
The project utilizes a suite of standard computer vision benchmark datasets designed to simulate a continuous stream of learning tasks:
 * CIFAR-100: Structured as 20 sequential tasks with 5 classes each.
 * Split Mini-ImageNet: Higher complexity images, structured as 20 tasks × 5 classes.
 * Permuted MNIST: 10 tasks where each task applies a fixed random permutation to the image pixels to simulate domain shifts.
 * Sequential Tiny-ImageNet: A large-scale evaluation set with 10 tasks × 20 classes.

# About Features
Since these are image datasets, the primary "features" are raw image pixel arrays (e.g., RGB matrices). During training, the neural network models automatically extract these into high-dimensional distributed representations (embeddings) across their hidden layers to distinguish between different classes and tasks.

# Which Preprocessing Algo
The preprocessing pipeline will include:
 * Image Resizing: Normalizing input dimensions to fit the base model architectures.
 * Pixel Normalization: Standardizing pixel values (typically scaling to a 0-1 or -1 to 1 range) to ensure stable gradient descent.
 * Task-Specific Transformations: Applying spatial pixel permutations specifically for the Permuted MNIST dataset to generate distinct task domains.

# Which Methodology
The project uses a Comparative Experimental Methodology. It will systematically evaluate multiple "Hybrid" continual learning methods—which combine different strategies like Experience Replay (storing old data), Knowledge Distillation (transferring model states), and Regularization (penalizing weight changes). The performance will be evaluated using specific metrics: Average Accuracy, Forgetting Measure, and Forward/Backward Transfer over multiple random seeds to ensure statistical significance.

# Flow Diagram (Write this description or draw it as boxes and arrows)
 * Data Stream Input: Sequence of new tasks (Task 1, Task 2... Task N) arrives over time.
 * Preprocessing Module: Images are resized and normalized.
 * Hybrid Continual Learning Model: * Path A: New data passes through the Neural Network for training.
   * Path B: A Replay Buffer retrieves a small batch of old task data.
 * Loss Calculation: Model computes loss using Cross-Entropy (new data) + Distillation/Regularization penalties (to protect old knowledge).
 * Evaluation Engine: Calculates metrics (Average Accuracy, Forgetting, etc.).
 * Decision Support Output: Maps resource constraints to the optimal algorithm.

# Which Algorithm
The project implements a combination of base architectures and Continual Learning (CL) algorithms:
 * Base Architectures: ResNet-18 (Convolutional) and ViT-Small (Vision Transformer).
 * CL Algorithms: Dark Experience Replay (DER/X-DER), iCaRL (Incremental Classifier and Representation Learning), and ER-Reservoir.

# What plan for front end
The front end will be a "Decision Support Framework" dashboard built using a Python web framework like Streamlit or Gradio. It will feature an interactive UI where users can input their specific hardware constraints (e.g., memory limits, compute availability) and task parameters. The application will dynamically process these inputs and display recommendations for the most optimal continual learning method to use.

# What is the expected output
The final expected outputs include:
 * Empirical performance tables and Pareto-optimality charts illustrating the stability-plasticity trade-off for each algorithm.
 * An interactive decision tree/flowchart that successfully recommends optimal hybrid CL methods based on user constraints.
 * A reproducible open-source code repository containing the evaluation framework.

