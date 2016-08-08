# Visual Categorization Task Model
This is a model of a task detailed here (http://science.sciencemag.org/content/291/5502/312.full), in which a monkey is shown a series of objects (specifically, dogs and cats) and is required to push a leaver if the most recent two objects belong to the same category (ie, both of the pictures were of the same type of animal). There’s another follow up paper that details the underlying neurobiology of the task, found here: http://cerco.ups-tlse.fr/pdf0609/thorpe_sj_01_260.pdf


It provides the following image, which contains typical responce latencies between different parts of the brain. ![alt tag](https://github.com/agaikova/visual-categorization/blob/master/latencies.png)


In order to evaluate the success of the model, we want the latencies of different brain structures in monkeys to match up with the latencies between analogous components of the model.


## Expected latency ranges

The delay between various brain components are as follows:

    retina_to_V1 = 40 ms
    V1_to_AIT = 40 ms
    AIT_to_PFC = 30 ms
    PFC_to_PMC = ~20ms
    PMC_to_MC = ~25 ms
    MC_to_finger = ~25 ms
    finger_to_end = ~ 30 ms

I calculated all these ranges by taking the average of the delay ranges and calculating differences between them (for example, the V1 latency would be 50ms and the AIT latency average would be 90ms, so the delay between V1 and AIT would be ~ 40 ms).

The model (found in the file called second_pass) was at least partially stolen from Jelmer.


## Data Analysis Stuff

Initially, I decided to group parameters relating to connection strengths in different components of the system together with the latencies that they would impact (vision parameters with vision-related latencies, etc) and run hyperopt on those. The progression of notebooks detailing those procedures are as follows:

1. Vision

2. Vision_plus

3. Motor

At this point, I did a final test using the parameters found in the 3 hyperopt aforementioned hyperopt notebooks. That can be found in Final_probe_test.ipynb (4). I realized that the initial value for the motor_transform parameter had been 10 but I had limited it to 0-1 in notebook #3, so I re-ran the optimization process for the motor system in a notebook called hyperopt_motor_2 (5). Then I re-evaluated the probes (final_probe_test_2(6)).


After all that, I decided to run hyperopt on each individual parameter. Details about that can be found in the “Individual Parameters” file. The last and final probe test, called Final_FINAL_test, reveals that we landed in an appropriate ballpark for all latency values
