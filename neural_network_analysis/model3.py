"""
Increase model capacity. Add more layers at the end.
Result:
Parameters: 11.9k
Best Train Accuracy: 99.33
Best Test Accuracy: 99.04
Analysis:
The model still showing over-fitting, possibly DropOut is not working as expected! Wait yes! We don't know which layer is causing over-fitting. Adding it to a specific layer wasn't a great idea.
Quite Possibly we need to add more capacity, especially at the end.
Closer analysis of MNIST can also reveal that just at RF of 5x5 we start to see patterns forming.
We can also increase the capacity of the model by adding a layer after GAP!
"""
