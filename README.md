# Visual Categorization Task Model
This is a model of a task detailed here (http://science.sciencemag.org/content/291/5502/312.full), in which a monkey is shown a series of objects (specifically, dogs and cats) and is required to push a lever if the most recent two objects belong to the same category (ie, both of the pictures were of the same type of animal). The researchers created their stimuli by generating dog/cat “morphs” from a list of 6 prototypical “dog” or “cat” images. Each “morph” would be a part dog and part cat, the more dominant of which would correspond to the category that the morph belonged to.  This means that their set of dog-like stimuli was composed of “morphs” that were more than 50% similar to one of their 3 prototypical dogs. The monkeys then performed a delayed match-to-category task in which they were presented with two stimuli and required to judge whether the stimuli were from the same category.  

There’s a follow up paper that details the underlying neurobiology of the task, found here: http://cerco.ups-tlse.fr/pdf0609/thorpe_sj_01_260.pdf. It examines the time delays in activation of different parts of the brain, which are well summarized by this image: ![alt tag](https://github.com/agaikova/visual-categorization/blob/master/latencies.png)

The model written for this task is composed of a visual system, a motor system, and intermediate compare modules. The visual system takes jpeg images of words and uses Gabor and Mask encoders (Written by Eric Hunsberger) to map them to vectors which represent the categories of concepts. The timing data that we are comparing our model to only reflects the categorization aspect of the task, so instead of using stimuli that were ambiguous I used two images of words. Those two words were arbitrarily chosen to be  “Whiskey” and “Fatigue”, and were presented one after the other. The vector corresponding to the presented stimuli is then compared to what is being stored in the working memory. If they are the same, the motor system decides to press the button. If not, the motor system stops presenting the press button vectors, so the model functionally categorizes the presented word and word in working memory as “same” or “different”. 

Since the nengo model has components analogous to the actual brain, we decided that we could compare the delay time response of the components of our model to the times discussed in the paper. The following are approximations in the expected delay of various components, which were calculated from the differences between the averages of the ranges given in the aforementioned image. 

		retina_to_V1 = 40 ms
		V1_to_AIT = 40 ms
		AIT_to_PFC = 30 ms
		PFC_to_PMC = ~20ms
		PMC_to_MC = ~25 ms
		MC_to_finger = ~25 ms
		finger_to_end = ~ 30 ms

In order to evaluate the success of the model, we want the latencies of different brain structures in monkeys to match up with the latencies between analogous components of the model. To get the activation times out of the model, each of the probed ensembles was compared to what it should ideally contain (for example - the decoded output of the V1 ensemble was compared to the image that was being presented to it). Once the similarity between a decoded output and the expected object was more than 0.5, that ensemble was considered to have been activated. The latencies were calculated by subtracting the activation times from the appropriate ensembles. 

Initially, the model ran too fast. In order to fit our measured latencies to the empirical data, the model would have to be slowed down. One way of doing that would be to inject feedback connections between ensembles and trying to optimize the strengths of said feedback connections so that the resulting latencies match up with biological data. Here’s a complete list of all the parameters considered variable: 
![alt tag](https://github.com/agaikova/visual-categorization/blob/master/networkimage.png)


		Result feedback
		Compare to result feedback
		Motor transform
		Motor feedback
		Finger feedback
		Motor to finger strength
		AIT to V1 strength
		V1 recursive connection
		AIT recursive connection

The list includes all recursive connections, transform scaling and feedback connections. 

In order to optimize the aforementioned parameters, I used a program called hyperopt (a tutorial of which you can find here[link]). Since there are so many, It made sense to split the parameters into 3 groups: one set of parameters related to the vision system, one related to the visual system, and one for the intermediate steps. 
I then ran preliminary hyperopt optimization on each of the three individual groups. This gave me a set of working values for the optimization tests I then ran on the  individual parameters. In said single-parameter optimizations, all parameters other than the one being tested were held constant at the values found by the multi-parameter optimization.

The progression of notebooks are as follows: 
Hyperopt Vision - Contains parameters associated with vision (AIT to V1 strength, V1 recursive connection, AIT recursive connection)
Hyperopt Vision plus - Runs code with the recursive connection strengths found in Vision notebook while changing the parameters of the result and compare modules
Hyperopt Motor - Runs code with the recursive connection strengths found in previous notebooks while varying motor system parameters
Final probe test - Tests the accuracy of the multi-parameter optimization
Hyperopt Motor 2 - I realized that the first motor test had a parameter range that was wrong, so I fixed that and re-ran this the motor optimization
Final probe test 2 - Tests the accuracy of the multi-parameter optimization for motor 2
All the individual tests (found in the folder called Individual Parameters)
Final FINAL test - Tests the accuracy of the single-parameter optimization

##Installation Instructions
To set up this model, git clone this repository (https://github.com/agaikova/visual-categorization.git). You’ll need to have the following list of dependencies installed:

		Nengo
		Matplotlib
		Nengo_extras (specifically the vision module)
		Pypng
		Itertools
		Base64
		PIL
		cStringIO
		And standard things like numpy, inspect, os, sys, time and csv
The model itself can be found in the file titled “second_pass.py”. 

