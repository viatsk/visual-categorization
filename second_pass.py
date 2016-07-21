import nengo
import nengo.spa as spa
#import nengo_ocl
import numpy as np
import inspect, os, sys, time, csv, random
import matplotlib.pyplot as plt
from nengo_extras.vision import Gabor, Mask

import png
import itertools
import base64
import PIL.Image
import cStringIO

from time import sleep
########### Specify what information is being fed into the model ###########
D = 32
vis_items = ['FATIGUE', 'WHISKEY']
vis_vocab = spa.Vocabulary(D)

result_vocab_items = ['SAME']
input_items = ['PUSH']
action_items = ['F1']

##### Vision and motor system #########
import vision_system as v
import motor_system as m
reload (v)
reload (m)

directory = '/home/stacy/github/visual-categorization/assoc_recog_s/images/'
image_list = v.load_images(directory, items = vis_items)
output_list = v.vector_gen_function(vis_items, vocab = vis_vocab)


### List of feeback things to change ###
result_feedback = 0.5
compare_to_result_strength = 0.1

### MODEL ###
model = spa.SPA(label = 'MAIN')
with model:
    model.vision_system = v.make_vision_system(image_list, output_list, n_neurons = 500)
    model.concept = spa.State(D, vocab = vis_vocab)
    nengo.Connection(model.vision_system.AIT, model.concept.input)

    model.compare = spa.Compare(D)
    model.wm = spa.State(D, vocab = vis_vocab)
    model.result = spa.State(D, feedback=result_feedback)
    
    
    nengo.Connection(model.concept.output, model.compare.inputA, synapse=0.01)
    nengo.Connection(model.wm.output, model.compare.inputB)
    
    vocab = model.get_input_vocab('result')
   
    #I like this function so I'm keeping it here. I think it is very beautiful.
    #def conn_func(vocab, things_to_add):
    #    for i in range(len(things_to_add)):
    #        new = vocab.parse(things_to_add[i]).v
    #    transform = 0.1*np.array([new]).T

    #    return 0.4*transform

    nengo.Connection(model.compare.output, model.result.input,
            #function = conn_func(vocab, result_vocab_items))
            transform=compare_to_result_strength*np.array([vocab.parse('SAME').v]).T)
    
    # Mapping from the vocab object with all the result items to the vocab object with all the things motor is expecting 
    # to get as input
    def result_to_motor(in_vocab, out_vocab):
        mapping = np.zeros((D,D))
        for i in range(len(input_items)):
            mapping += np.outer(in_vocab.parse(result_vocab_items[i]).v, out_vocab.parse(input_items[i]).v)
        transform = mapping.T
        return transform
    
    model.motor_system = m.make_motor_system(input_items, action_items)

    nengo.Connection(model.result.output, model.motor_system.motor_input.input, 
        transform = result_to_motor(vocab, model.motor_system.motor_vocab))
    
    def present_func(t):
        if t < 1:
           index = 0
        else:
            index = 1
        #index = int(t/0.1)
        #return image_list[index % len(image_list)]
        return image_list[index]
    
    
    stim = nengo.Node(present_func)
    nengo.Connection(stim, model.vision_system.presentation_node)

    stim_wm = nengo.Node(model.get_input_vocab('wm').parse('FATIGUE').v)
    nengo.Connection(stim_wm, model.wm.input)
    
    # This code makes several objects in this model accessible
    
    V1_probe = nengo.Probe(model.vision_system.V1)
    AIT_probe = nengo.Probe(model.vision_system.AIT, synapse = 0.005)

    PFC_probe = nengo.Probe(model.compare.output, synapse = 0.005)
    PMC_probe = nengo.Probe(model.result.output, synapse = 0.005)
    MC_probe = nengo.Probe(model.motor_system.motor.output, synapse = 0.005)
    finger_probe = nengo.Probe(model.motor_system.fingers.output, synapse = 0.005)
    
    final_probe = nengo.Probe(model.motor_system.finger_pos.output, synapse = 0.005)

