# brain-signal-interpreter

## Backstory
This was a research project I did the spring semester of my junior year (2023) at Clemson. I worked with another student from the physics department who created a headset to scan electroencephalography (EEG) signals from the brain. Our goal was to create our own neural network to accurately predict movements of body solely from the EEG readings of the headset.

We gathered hundreds of short data samples that either contained blinks or no blinks, and were labeled accordingly. This binary process (blink or no-blink) made for a very straightforward approach to data collection and allowed us to test proof of concept before moving to more complex applications. 

## Results
Through carefully transforming our data and optimizing our TensorFlow convolutional neural network, we were able to train a model to predict human blinks with over a 99% success rate.

## Future applications
Although the project has been paused, there were a few directions we dicussed we could go with the project. Since the neural network was much better than expected at predicted blinks, we speculated that we could make a program for disabled people to type in morse code with their blinks while using the headset. Another step forward was to train a new neural network to predict certain finger/hand movements which could aid prostetics. The main limiting factor for continuing with one of these visions was the extremely time-consumely data collection that would need to take place for either of these to be accomplished.

## How to run
Requirements:
* Python3
* TensorFlow
* Numpy
* Pickle

1. Enter the pathnames of your blink and no_blink data samples into the DataReader.py parameters and adjust the pickle names accordingly. Run the script (python3 DataReader.py).
2. Repeat previous step for the test data you will use to test the model after training.
3. Adjust the pickle pathnames for BlinkNetwork.py in lines 14-15 and 54-55. Run the BlinkNetwork script to see the results (python3 BlinkNetwork.py).
