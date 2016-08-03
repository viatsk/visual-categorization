import nengo
import nengo.spa as spa
#import nengo_ocl
import numpy as np
import inspect, os, sys, time, csv, random
from nengo_extras.vision import Gabor, Mask

import png
import itertools
import base64
import PIL.Image
import cStringIO

import ctn_benchmark

class Vision_Model(ctn_benchmark.Benchmark):
    def params(self):
        self.default('runtime', runtime = 2)
        self.default('result_feedback', result_feedback = 0.46003873368799186)
        self.default('compare_to_result_strength', compare_to_result_strength = 0.1416302690767407)
        self.default('D', D = 32)
        
        # THINGS COMING FROM MOTOR
        self.default('motor_feedback', motor_feedback = 0.13055354933808305)
        self.default('motor_transform', motor_transform = 2.0)
        self.default('finger_feedback', finger_feedback = 0.6964691855978616)
        self.default('motor_to_fingers_strength', motor_to_fingers_strength = 0.8333928248007452)
        
        # THINGS COMING FROM VISION
        self.default('AIT_V1_strength', AIT_V1_strength = 0.10606490595473272)
        self.default('V1_r_transform', V1_r_transform = 0.10606490595473272)
        self.default('AIT_r_transform', AIT_r_transform = 0.6964691855978616)
    
    def model(self, p):
        vis_items = ['FATIGUE', 'WHISKEY']
        vis_vocab = spa.Vocabulary(p.D)
        self.vis_items = vis_items
        self.vis_vocab = vis_vocab

        result_vocab_items = ['SAME']
        input_items = ['PUSH']
        action_items = ['F1']
        self.result_vocab_items = result_vocab_items
        self.input_items = input_items
        self.action_items = action_items
        
        ##### Vision and motor system #########
        import vision_system as v
        import motor_system as m
        reload (v)
        reload (m)

        directory = '/home/stacy/github/visual-categorization/assoc_recog_s/images/'
        image_list = v.load_images(directory, items = vis_items)
        output_list = v.vector_gen_function(vis_items, vocab = vis_vocab)
        self.directory = directory
        self.image_list = image_list
        self.output_list = output_list
        
        model = spa.SPA(label = 'MAIN')
        with model:
            model.vision_system = v.make_vision_system(image_list, output_list, n_neurons = 500, AIT_V1_strength = p.AIT_V1_strength, AIT_r_transform = p.AIT_r_transform, V1_r_transform = p.V1_r_transform)
            model.concept = spa.State(p.D, vocab = vis_vocab)
            nengo.Connection(model.vision_system.AIT, model.concept.input)

            model.compare = spa.Compare(p.D)
            model.wm = spa.State(p.D, vocab = vis_vocab)
            model.result = spa.State(p.D, feedback=p.result_feedback)
            
            
            nengo.Connection(model.concept.output, model.compare.inputA, synapse=0.01)
            nengo.Connection(model.wm.output, model.compare.inputB)
            
            vocab = model.get_input_vocab('result')
            nengo.Connection(model.compare.output, model.result.input,
                        transform=p.compare_to_result_strength*np.array([vocab.parse('SAME').v]).T)
            
            def result_to_motor(in_vocab, out_vocab):
                mapping = np.zeros((p.D,p.D))
                for i in range(len(input_items)):
                    mapping += np.outer(in_vocab.parse(result_vocab_items[i]).v, out_vocab.parse(input_items[i]).v)
                transform = mapping.T
                return transform
    
            model.motor_system = m.make_motor_system(input_items, action_items, motor_feedback=p.motor_feedback, motor_transform = p.motor_transform, finger_feedback = p.finger_feedback, motor_to_fingers_strength = p.motor_to_fingers_strength)

            nengo.Connection(model.result.output, model.motor_system.motor_input.input, 
                transform = result_to_motor(vocab, model.motor_system.motor_vocab))
            def present_func(t):
                if t < 1:
                   index = 0
                else:
                    index = 1
                return image_list[index]
            
            stim = nengo.Node(present_func)
            nengo.Connection(stim, model.vision_system.presentation_node)

            stim_wm = nengo.Node(model.get_input_vocab('wm').parse('FATIGUE').v)
            nengo.Connection(stim_wm, model.wm.input)
            
            self.V1_probe = nengo.Probe(model.vision_system.V1)
            self.AIT_probe = nengo.Probe(model.vision_system.AIT, synapse = 0.005)

            self.PFC_probe = nengo.Probe(model.compare.output, synapse = 0.005)
            self.PMC_probe = nengo.Probe(model.result.output, synapse = 0.005)
            self.MC_probe = nengo.Probe(model.motor_system.motor.output, synapse = 0.005)
            self.finger_probe = nengo.Probe(model.motor_system.fingers.output, synapse = 0.005)
            
            self.final_probe = nengo.Probe(model.motor_system.finger_pos.output, synapse = 0.005)
            self.mymodel = model
        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.runtime)
        
        def get_delay(probe, object_to_compare, V1 = False):
            data = sim.data[probe]
            dot_prod = np.dot(data, object_to_compare)
            if V1:
                lat = np.where(dot_prod >= ((dot_prod.max() + dot_prod.min())/2))[0]
            else: 
                lat = np.where(dot_prod >= dot_prod.max()/2)[0]
            start, end = lat[0], lat[-1]
            return sim.trange()[start], sim.trange()[end]
            
        V1_start, V1_end = get_delay(self.V1_probe, self.image_list[0], V1 = True)
        AIT_start, AIT_end = get_delay(self.AIT_probe, self.mymodel.get_output_vocab('wm').parse('FATIGUE').v)
        PFC_start, PFC_end = get_delay(self.PFC_probe, np.ones((1, 2000)))
        PMC_start, PMC_end = get_delay(self.PMC_probe, self.mymodel.get_output_vocab('result').parse('SAME').v)
        MC_start, MC_end = get_delay(self.MC_probe, self.mymodel.motor_system.motor_vocab.parse('PUSH').v)
        finger_start, finger_end = get_delay(self.finger_probe, self.mymodel.motor_system.finger_vocab.parse('F1').v)
        final_start, final_end = get_delay(self.final_probe, np.ones((1, 2000)))
        
        retina_to_V1 = V1_start
        V1_to_AIT = AIT_start - V1_start
        AIT_to_PFC = PFC_start - AIT_start
        PFC_to_PMC = PMC_start - PFC_start
        PMC_to_MC = MC_start - PMC_start
        MC_to_finger = finger_start - MC_start
        finger_to_end = final_start - finger_start
        
        return {'retina_to_V1':retina_to_V1, 'V1_to_AIT':V1_to_AIT, 'AIT_to_PFC':AIT_to_PFC, 'PFC_to_PMC':PFC_to_PMC, 'PMC_to_MC':PMC_to_MC, 'MC_to_finger':MC_to_finger, 'finger_to_end':finger_to_end}
        
