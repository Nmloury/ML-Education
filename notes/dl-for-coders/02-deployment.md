# Lesson 2 - Deployment

- how do we put a model into production?
  - find your problem
  - get your data
  - clean your data
    - before you clean your data, you train a model! This seems pretty counter intuitive but will make sense

## Training the Model
- data augmentation
  - changing, resizing, adjusting, slightly cropping an image can make a model more robust because the label stays the same
  - especially useful when training for more than 5 epochs
  - Confusion matrix
    - useful for cateogorical data
    - what error is the model making by comparing the actual label to the predicted label
  - looking at the the examples with the highest loss can give us insight into when the model is the most wrong
  - Before you starting cleaning, ALWAYS builed a model to see what things are difficult to recognize in your data
    - the model can also help you find data problems as well
    - can be helpful for figuring out how to gather better data as well as help with the cleaning process

## Putting Into Production
- HuggingFace Spaces is a great way to deploy an ML model with Gradio (there's a blog post I can look at to see how to do this)
- we will copy our model to the HuggingFace Spaces server and then create a user interface for it
- very easy to create a space
- use apache licenses
- creating a space with create a github repo
  - we will build our huggingface spaces/gradio app app using that repo
  - now the question is how do we get the model into production?
    - we can export our trained model by pickling !
- how do we do predictions on a saved model?
  - we will need any functions that we used to create the model available in the file we are using for predictions
  - `load_learner()` to load our model into the file
  -  we want to create a gradio function that will call our model
  -  we can use `from nbdev.export import notebook2script` to turn our jupyter notebook into a python script! (this is what gradio expects for our app)
  -  then git push to put app.py onto our repo and boom!
-  huggingface spaces also creates an api endpoint that we can then call within any application (i.e. one built with javascript)
-  we can create an app like this on github pages


## Jupyter Notebook Tricks/Tips
- if you put a ? before a function in jupyter notebook you'll get a short bit of information about the function
  - ?? will give you the full source code
  - `doc()` will give you a link the the full documentation
  - 

