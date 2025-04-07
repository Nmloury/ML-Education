# Deep Learning Flashcards - Chapter 1

## General Deep Learning Myths and Reality

Q: What are the common myths about deep learning?
A: The common myths are:
- Requires lots of math
- Needs lots of data
- Requires expensive computers
- Needs a PhD

Q: What is the reality about deep learning requirements?
A: The reality is:
- Just high school math is sufficient
- Record-breaking results possible with <50 items
- State-of-the-art work possible with free resources
- No PhD required

## Key Applications

Q: What are the main areas where deep learning excels?
A: Deep learning excels in:
- Natural Language Processing (NLP)
- Computer Vision
- Medicine
- Biology
- Image Generation
- Recommendation Systems
- Gaming
- Robotics

Q: What are specific applications in Natural Language Processing?
A: NLP applications include:
- Question answering
- Speech recognition
- Document summarization
- Text classification

Q: What are specific applications in Computer Vision?
A: Computer Vision applications include:
- Image recognition
- Object detection
- Segmentation

## Neural Networks History

Q: Who developed the first mathematical model of an artificial neuron and when?
A: Warren McCulloch and Walter Pitts in 1943 developed the first mathematical model of an artificial neuron.

Q: What was the Mark I Perceptron and who developed it?
A: The Mark I Perceptron was the first device based on artificial neuron principles, developed by Frank Rosenblatt in the 1950s.

Q: What was the significance of Minsky and Papert's "Perceptrons" book?
A: Their 1969 book showed limitations of single-layer perceptrons and demonstrated potential of multiple layers, leading to a decline in neural network research.

Q: What were the key requirements for parallel distributed processing (PDP) defined by Rumelhart and McClellan?
A: The eight requirements are:
1. Processing units
2. State of activation
3. Output function
4. Pattern of connectivity
5. Propagation rule
6. Activation rule
7. Learning rule
8. Operating environment

## Learning Deep Learning

Q: What are the key principles for choosing deep learning projects?
A: Key principles include:
- Choose projects you can start quickly
- Focus on accessible data
- Pick projects you understand
- Look for similar projects as templates
- Start small and iterate

Q: What are the main software tools for deep learning?
A: Main tools include:
- PyTorch (deep learning library)
- fastai (built on PyTorch)
- Jupyter (for data science)

## Image Recognition Model

Q: What are the key components of the image recognizer?
A: Key components include:
- DataLoaders (data batching and augmentation)
- Learner (training process management)
- Callbacks (progress monitoring)
- Metrics (performance tracking)

Q: What is the training process for the image recognizer?
A: The process includes:
1. Data preparation and augmentation
2. Model initialization with pretrained weights
3. Fine-tuning of final layers
4. Training with progressive unfreezing
5. Validation and performance monitoring

Q: What are the performance optimization techniques used?
A: Optimization techniques include:
- GPU acceleration
- Mixed precision training
- Gradient clipping
- Early stopping

## Deep Learning Applications Beyond Images

Q: What are the main types of deep learning applications beyond image classification?
A: Main types include:
- Text Classification
- Tabular Data Analysis
- Recommendation Systems
- Time Series Analysis
- Natural Language Processing
- Audio Processing
- Multi-Modal Applications
- Domain-Specific Applications

Q: What are the key advantages of deep learning across different domains?
A: Key advantages include:
- Automatic feature learning
- Scalability to large datasets
- Transfer learning capabilities
- End-to-end training
- Continuous improvement with more data

## Technical Concepts

Q: What is the difference between classification and regression?
A: Classification predicts discrete categories (e.g., cat vs dog), while regression predicts continuous values (e.g., house prices).

Q: What is overfitting and how can it be identified?
A: Overfitting occurs when a model memorizes training data instead of learning general patterns. It can be identified when a model performs perfectly on training data but fails on new data.

Q: What is the difference between a metric and loss?
A: A metric is a human-readable measure of model performance, while loss is the function used by the training algorithm to update weights.

Q: What are hyperparameters?
A: Hyperparameters are parameters that control the training process and are not learned during training (e.g., learning rate, number of layers).

Q: What is the "head" of a model?
A: The "head" refers to the final layers of a model that are customized for specific tasks, often replaced when fine-tuning pretrained models.

Q: What features do early vs later layers of a CNN find?
A: Early layers find basic features (edges, gradients, colors), while later layers find complex features (shapes, objects, patterns).

## Best Practices

Q: What's the best way to avoid failures when using AI in an organization?
A: Best practices include:
- Start with small, well-defined projects
- Focus on practical applications
- Ensure proper data quality and validation
- Monitor model performance and biases
- Maintain clear communication with stakeholders

Q: What is the Universal Approximation Theorem?
A: It's a theorem showing that a neural network can solve any mathematical problem to any level of accuracy.

Q: What are the essential components needed to train a model?
A: Essential components include:
- Training data
- Model architecture
- Loss function
- Optimization algorithm
- Performance evaluation method 