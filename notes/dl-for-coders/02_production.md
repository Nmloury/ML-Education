# From Model to Production

## Summary

This chapter covers the end-to-end process of creating a deep learning application, using a bear classifier as an example. Key topics include:

- Understanding the capabilities and constraints of deep learning
- Gathering and preparing data for model training
- Training a model and using it to clean data
- Deploying a model as a web application
- Avoiding potential disasters in production

The chapter emphasizes practical approaches to deep learning, focusing on iterative development and careful deployment strategies. It introduces the Drivetrain Approach for designing data products and discusses common challenges like out-of-domain data and domain shift.

## The Practice of Deep Learning

### Starting Your Project

- Choose projects where you can get started quickly with available data
- Iterate from end to end rather than perfecting individual components
- Focus on practical experience and learning by doing
- Start with small, well-defined projects that you understand
- Look for similar projects as templates

### The State of Deep Learning

#### Computer Vision
- Deep learning excels at object recognition, detection, and segmentation
- Models may struggle with images significantly different from training data
- Data augmentation can help improve model robustness
- Image labeling can be slow and expensive

#### Text (Natural Language Processing)
- Good at classification tasks (spam, sentiment, author, etc.)
- Excellent at generating context-appropriate text
- Not reliable for generating factually correct responses
- Concerns about disinformation and misuse

#### Combining Text and Images
- Deep learning can effectively combine text and images
- Models can generate captions for images
- No guarantee of factual correctness
- Recommended as part of a human-in-the-loop process

#### Tabular Data
- Recent improvements in handling time series and tabular data
- Often used as part of an ensemble with other models
- Can handle a wider variety of column types
- Generally takes longer to train than traditional methods

#### Recommendation Systems
- Special type of tabular data with high-cardinality categorical variables
- Good at handling high-cardinality categorical variables
- May recommend products users already know about
- Focus on what users might like rather than what would be helpful

#### Other Data Types
- Domain-specific data can often fit into existing categories
- Protein chains can be treated like natural language
- Sounds can be represented as spectrograms and treated as images

### The Drivetrain Approach

The Drivetrain Approach is a method for designing data products with these steps:

1. Define a clear objective
2. Identify levers (actions you can take)
3. Determine what data you need
4. Build predictive models

Example with recommendation systems:
- Objective: Drive additional sales through surprising recommendations
- Lever: Ranking of recommendations
- Data: Results of randomized experiments
- Models: Purchase probabilities conditional on seeing/not seeing recommendations

## Gathering Data

### Using Bing Image Search
- Free for up to 1,000 queries per month (up to 150 images per query)
- Requires an Azure account and API key
- Results can vary and may contain unexpected content

### Data Cleaning
- Verify downloaded images to remove corrupt files
- Use the model to help identify mislabeled data
- Clean data iteratively rather than before training

## From Data to DataLoaders

### Creating DataLoaders
- DataLoaders is a class that stores DataLoader objects
- To create DataLoaders, we need to specify:
  1. Types of data (blocks)
  2. How to get the list of items (get_items)
  3. How to label these items (get_y)
  4. How to create the validation set (splitter)

### Data Augmentation
- Creates random variations of input data
- Common techniques include rotation, flipping, perspective warping, brightness changes
- Helps models learn to recognize objects in different positions and sizes
- Can be applied to individual items (item_tfms) or batches (batch_tfms)

## Training Your Model, and Using It to Clean Your Data

### Training Process
- Use RandomResizedCrop for image resizing
- Apply data augmentation with aug_transforms
- Fine-tune a pretrained model (ResNet18)
- Monitor performance with metrics like error_rate

### Model Evaluation
- Use confusion matrix to visualize model performance
- Identify high-loss images to find potential data issues
- Use ImageClassifierCleaner to clean data
- Retrain model after cleaning

## Turning Your Model into an Online Application

### Using the Model for Inference
- Save model with export() method
- Load model with load_learner()
- Use predict() method for inference
- Access model predictions and probabilities

### Creating a Notebook App
- Use IPython widgets (ipywidgets) for GUI components
- Create file upload, image display, and prediction widgets
- Connect widgets with event handlers
- Arrange widgets in a vertical box (VBox)

### Deploying Your App
- Voilà converts Jupyter notebooks to web applications
- CPU is often sufficient for inference (no GPU needed)
- Options for deployment include Binder (free)
- Consider server vs. edge device deployment

## How to Avoid Disaster

### Deployment Process
1. Start with manual process, running model in parallel
2. Limit scope and carefully supervise
3. Gradually increase scope with good reporting systems

### Potential Issues
- Out-of-domain data: Data different from training data
- Domain shift: Data changes over time
- Unforeseen consequences and feedback loops
- Bias amplification in systems like predictive policing

### Best Practices
- Consider "What would happen if it went really, really well?"
- Implement careful rollout plans with monitoring
- Ensure reliable communication channels for issues
- Maintain human oversight of automated systems

## Get Writing!

- Writing helps solidify understanding
- Blog about your deep learning journey
- Share your experiences and insights
- Help people one step behind you

## Questionnaire

1. **Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.**
   - The model might struggle with nighttime images, low-resolution camera images, or bears in unusual positions (from behind, partially covered by bushes, or far from the camera).

2. **Where do text models currently have a major deficiency?**
   - Text models are not good at generating factually correct responses. There's no reliable way to combine a knowledge base with a deep learning model for generating medically correct or factually accurate natural language responses.
    
3. **What are possible negative societal implications of text generation models?**
   - They could be used to spread disinformation, create unrest, and encourage conflict at massive scale. They can generate compelling but entirely incorrect content that appears credible to laypeople.

4. **In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?**
   - Use the model as part of a human-in-the-loop process where humans verify and act on the model's outputs. This can make humans more productive while maintaining accuracy.

5. **What kind of tabular data is deep learning particularly good at?**
   - Deep learning is good at handling high-cardinality categorical columns and columns containing natural language (book titles, reviews, etc.).

6. **What's a key downside of directly using a deep learning model for recommendation systems?**
   - They only tell you what products a user might like, rather than what recommendations would be helpful. They might recommend products users already know about or different packaging of products they already have.

7. **What are the steps of the Drivetrain Approach?**
   1. Define a clear objective
   2. Identify levers (actions you can take)
   3. Determine what data you need
   4. Build predictive models

8. **How do the steps of the Drivetrain Approach map to a recommendation system?**
   - Objective: Drive additional sales through surprising recommendations
   - Lever: Ranking of recommendations
   - Data: Results of randomized experiments
   - Models: Purchase probabilities conditional on seeing/not seeing recommendations

9. **Create an image recognition model using data you curate, and deploy it on the web.**
   - This is a practical exercise that involves gathering data, training a model, and deploying it using tools like Voilà and Binder.

10. **What is DataLoaders?**
    - A fastai class that stores multiple DataLoader objects, typically train and valid, making them available as properties.

11. **What four things do we need to tell fastai to create DataLoaders?**
    1. Types of data (blocks)
    2. How to get the list of items (get_items)
    3. How to label these items (get_y)
    4. How to create the validation set (splitter)

12. **What does the splitter parameter to DataBlock do?**
    - It determines how to split the data into training and validation sets, such as using RandomSplitter to randomly divide the data.

13. **How do we ensure a random split always gives the same validation set?**
    - By setting a seed parameter in the RandomSplitter, which ensures the same starting point for the random number generator.

14. **What letters are often used to signify the independent and dependent variables?**
    - x for independent variables and y for dependent variables.

15. **What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?**
    - Crop: Cuts the image to fit a square shape, potentially losing important details
    - Pad: Adds zeros (black) around the image, wasting computation
    - Squish: Stretches or compresses the image, creating unrealistic shapes
    - RandomResizedCrop is often preferred as it randomly selects parts of the image, helping the model learn to recognize objects in different positions.

16. **What is data augmentation? Why is it needed?**
    - Data augmentation creates random variations of input data while preserving meaning. It's needed to help models learn to recognize objects in different positions, sizes, and lighting conditions, making them more robust.

17. **What is the difference between item_tfms and batch_tfms?**
    - item_tfms are applied to individual items, while batch_tfms are applied to entire batches of data at once, which can be more efficient when using GPU.

18. **What is a confusion matrix?**
    - A visualization that shows the relationship between predicted and actual labels, with rows representing actual categories and columns representing predicted categories.

19. **What does export save?**
    - It saves both the model architecture and trained parameters, along with the definition of how to create the DataLoaders.

20. **What is it called when we use a model for getting predictions, instead of training?**
    - Inference

21. **What are IPython widgets?**
    - GUI components that bring together JavaScript and Python functionality in a web browser, allowing interactive elements in Jupyter notebooks.

22. **When might you want to use CPU for deployment? When might GPU be better?**
    - CPU is often better for inference when processing one item at a time, as GPUs are most efficient with parallel processing of many items.
    - GPU might be better for high-volume batch processing or when latency is critical and you have enough work to keep the GPU busy.

23. **What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?**
    - Requires network connection and adds latency
    - Privacy concerns if sensitive data is sent to a remote server
    - Managing server complexity and scaling
    - Higher operational costs

24. **What are three examples of problems that could occur when rolling out a bear warning system in practice?**
    - Working with video data instead of static images
    - Handling nighttime images not in the training data
    - Dealing with low-resolution camera images
    - Ensuring results are returned fast enough to be useful
    - Recognizing bears in positions rarely seen in online photos

25. **What is "out-of-domain data"?**
    - Data that the model sees in production which is very different from what it saw during training.

26. **What is "domain shift"?**
    - When the type of data that the model sees changes over time, making the original training data less relevant.

27. **What are the three steps in the deployment process?**
    1. Start with manual process, running model in parallel
    2. Limit scope and carefully supervise
    3. Gradually increase scope with good reporting systems 