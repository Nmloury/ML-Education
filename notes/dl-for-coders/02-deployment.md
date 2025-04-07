# Deployment - From Model To Production
[link to chapter](https://github.com/fastai/fastbook/blob/master/02_production.ipynb)

## The Practice of Deep Learning

- Keep an open mind about the potential of applying dl to your problem of hocie
- Picking a project
  - most important thing is to have a project to work on --> thats how you gain real experience
  - you want something you can start quickly --> try to choose a project where data is relatively accessible
    - if you spend too long trying to find data, you'll never get started!
  - iterate from end to end on your project. Don't spend too long on any one step, get something that works and then you can worry about improving it later on
    - building end-to-end will also give you ideas about where to improve based on the end result
      - do you need more data to make your model accurate? You won't know until you try!
    - help with buy-in as well, everyone loves a working prototype
  - pick a project that you know something about --> makes data accessbility easier
  - look for similar projects as template for what you're trying to do --> no need to start from scratch and will help guide you throughout the project

## Deep Learning Applications (outdated by now)

### Computer Vision

- **object recognition** - what is in that image?
- **object detection** - where is the object in the image?
- **segmentation** - categorize each pixel in an image based on what kind of object it is
- out-of-domain data: data that was not including in the training set (i.e. black and white images if only color images were used for training)
  - not surprisingly, models generally don't fare well on out-of-domain data
- one major challenge for object detection systems is image labelling is slow and expensive
  - one apporach to let here is synthetic variations of input images (rotation, changing brightness/contrast) -- this is known as **data augmentation**
- A problem may not look like a computer vision problem, but sometimes it can become one with a little ingenuity (i.e. categorizing sounds based on an image of their waveform)

### Text (NLP)

- DL is good at classifying documents (spam detection, sentiment analysis, etc.)
- Good at text generation (see modern LLMs)
- Translation
- Document Summarization

### Combining text and images

- input = image, output = caption is a good way to train a DL model to create captions
- can be hard to tell if those captions are actually "correct"
  - good to have a human-in-the-loop for verification

### Tabular data

- for tabular data DL has been making great strides, but is often best used as part of an ensemble approach with other model types (i.e. gradient boosted trees)
- dl does allow for more types of columns to be included (natural language, high-cardinality categorical columns)

### Recommendation Systems

- Really just a special type of tabular data
  - generally have a high-cardinality categorical variable for users and another one for products (or songs, movies etc.)
- This creates a huge sparse matrix representing all users and products, the trick is filling in the empty spots
  - generally use **collaborative filtering** = what else did users that liked the product you like buy?

## The Drivetrain Approach

- Consider your objective first --> figure out actions you can take to meet that objective and what data you have that will help --> build a model you can use to determine the best actions to take to get the best results
- we are using data to produce **actionable outcomes**
- This will be an iterative process as you discover more about the problem at hand, but never lose sight of the ultimate objective once you start getting in the weeds

## Useful Jupyter Notebooks Functions

- typing `??function` will show you the documentation on the function in question
  - `?function` will give you a shortened version of the same
- pressing shift+tab while inside the parentheses of a function will display a window with the signature and a short description
  - press multiple times for more info
- typing `%debug%` in the cell after an error will open the python debugger which will let you inspect the content of every variable

## Gathering Data

- The internet is your friend
- Prebuilt datasets are your friend
- search apis (particularly duckduckgo) is your friend
- check your data after you've gathered as the world is full of bias or inappropriate data --> your model will not let you know!

### From Data to DataLoaders

- `DataLoaders` is a thin class that stores whatever `DataLoader` object you pass to it and makes them available as `train` and `valid`
  - this will provide data for our model
- `DataLoaders` needs 4 things
  - what data are we working with?
  - how to get the list of items?
  - how to label these items?
  - how to create the validation set?
- `DataBlock` has inputs for each for these specifications (see documentation or textbook for details)
  - a `DataBlock` is essentially a template for creating a `DataLoaders` object
- The `DataLoader` will provide btaches of items at a time to the GPU for training
- In practice, we will often want to randomly select parts of an image, crop to just that and then trains the model on the subset of each image
  - training the network with examples where the object is in slightly different place and different sizes helps it to understand what an object is and how it can be represented

### Data Augmentation

- **Data Augmentation** means creating random variations of our input data to make them appear different
- common examples
  - rotation
  - flipping
  - perspective warping
  - brightness changes
  - contrast changes
  - cropping
  - zooming
  - color jittering
  - adding noise
  - blurring
  - sharpening
  - elastic transformations
  - random erasing/cutout
  - mixup (blending two images)
  - grayscale conversion