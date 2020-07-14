# Overview of the approaches
Here you are able to read a short description about the two different training files and how to use the different models. Please note that you have to use a file with pretrained word embeddings if you want to run the training.py file of model_0 or model_2. This files supports the training and leads to usable results after a smaller ammount of training epochs. Therefor you have to download [this file](https://www.kaggle.com/watts2/glove6b50dtxt/download) and put it into the chatbot/data directory.

## Model 0
This model is created by connecting the different layers manually. This did not work well as we were not able to connect the layers the right way after a long debugging session. The result is that the model looks like it is training based on the data but it actually does not train any weights. That is why we are not able to provide a working model here. Anyway we want to show this approach to follow the development of the project. 

As this approach did not work, the code still looks very similar to the paper it is based on (There was no additional benefit on improving and granularly commenting the code)

Cronologically this model was made and used after model_1. But we ended up with the opinion to store it as model_0 because it did not led to any usable results.

This model was inspired by a [publication of Akira Takezawa](https://towardsdatascience.com/how-to-implement-seq2seq-lstm-model-in-keras-shortcutnlp-6f355f3e5639).

## Model 1
While training this model, we reached negative probabilities. So we had to stop th training after epoche 29. More informations about this approach are documented in the project report.
If you want to try using the model anyway, you have to download it from [here](https://drive.google.com/file/d/1-Wye2qLMIkrWpGFL0dcdSIaJVQuqD5T2/view?usp=sharing) into the directory "model1" and start the file "use_chatbot.py" in this directory. Unfortunately the model was too large to store in github.

"idxm.npy", "idxr.npy" and "metadate.pkl" are the results of the preprocessing.

As this approach is not the offical approach we are handing in as the result of our project, this file is not completely cleaned up. We just want to provide the opprtuinity to follow the steps we did in the development of the project.

This model was inspired by a [repository of tensorlayer](https://github.com/tensorlayer/seq2seq-chatbot).

## Model 2
Model 2 is our offical approach wich is also implemented in the frontend. To use this model you have to download it from [here](https://drive.google.com/drive/folders/1qkqUJqsTw3lYPvoIKi1xJLhdjgPWzu9b?usp=sharing), put the 3 files into the following directory: model_2/training/model and start the frontend as discribed [here](https://github.com/SimonScapan/AuntElisa).

Additional information about the functionality and the developemnt of this model are documented in the project report.

As this approach is the offical approach we are handing in as the result of our project, this file is cleaned up and structured in the different steps of development.

This model was inspired by a [repository of Moein Hasani](https://github.com/Moeinh77/Chatbot-with-TensorFlow-and-Keras).

