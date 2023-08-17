# brain-signal-interpreter

## Backstory
This was a research project I did the spring semester of my junior year (2023) at Clemson. I worked with another student from the physics department who created a headset to scan electroencephalography (eeg) signals from the brain. Our goal was to create our own neural network to accurately predict movements of body solely from the eeg readings of the headset.

We gathered hundreds of short data samples that either contained blinks or no blinks, and were labeled accordingly. This binary process (blink or no-blink) made for a very straightforward approach to data collection and allowed us to test proof of concept before moving to more complex applications. 

## Results
After training our model with some slight optimization changes, we were extremely excited to see that the model was able to predict whether or not a sample contained a blink with over a 99% success rate.
