import nengo
from nengo import spa
import numpy as np

D = 32
# INPUT VOCAB IS WHAT ITS GETTING FROM THE RESULT BOX
input_vocab = spa.Vocabulary(D) 
input_items = ['PUSH']

# ACTION ITEMS ARE THE THINGS IT"LL POTENTIALLY OUTPUT
action_vocab = spa.Vocabulary(D)
action_items = ['F1']


finger_strength = 0.55

def load_actions(vocab, items):
    possible_actions = []
    for i in items:
        n = vocab.parse(i).v
        possible_actions.append(n)
    return np.array(possible_actions)

    
def make_motor_system(input_items, action_items, motor_feedback = 0, motor_transform = 10, finger_feedback = 0.3, motor_to_fingers_strength = 0.4):
    motor_vocab = spa.Vocabulary(D)
    for item in range(len(input_items)):
        motor_vocab.parse(input_items[item])
        
    finger_vocab = spa.Vocabulary(D)
    for item in range(len(action_items)):
        finger_vocab.parse(action_items[item])
    
    motor_mapping = np.zeros((D,D))
    for item in range(len(input_items)):
        motor_mapping += np.outer(motor_vocab.parse(input_items[item]).v, finger_vocab.parse(action_items[item]).v).T
    
    motor_system = nengo.Network(label = 'motor_system')
    with motor_system:
        #input multiplier
        motor_input = spa.State(D,vocab=motor_vocab, label = 'motor_input')
    
        #higher motor area (SMA?)
        motor = spa.State(D, vocab = motor_vocab, feedback = motor_feedback, label = 'motor') # This thing has memory
        
        #connect input multiplier with higher motor area
        nengo.Connection(motor_input.output, motor.input,synapse=0.01, transform=motor_transform)
        
        #finger area
        fingers = spa.AssociativeMemory(finger_vocab, input_keys= action_items, wta_output=True, label = 'fingers')
    
        #conncetion between higher order area (hand, finger), to lower area (the actual motor mapping of the fingers)
        nengo.Connection(motor.output, fingers.input, transform=motor_to_fingers_strength*motor_mapping)
    
        #finger position - mapping to the actual finger movements! 
        finger_pos = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=len(input_items), label = 'finger_pos')
        nengo.Connection(finger_pos.output, finger_pos.input, synapse=0.1, transform=finger_feedback) #feedback
    
        #connection between finger area and finger position
        nengo.Connection(fingers.am.elem_output, finger_pos.input, transform = finger_strength) #fixed values, ideally
    
        #make things accessable
        motor_system.fingers = fingers
        motor_system.motor_input = motor_input
        motor_system.finger_pos = finger_pos
        motor_system.motor_vocab = motor_vocab
        motor_system.motor = motor
        motor_system.finger_vocab = finger_vocab 
        
    return motor_system


if __name__ == "__builtin__" or __name__ == 'builtins': 

    model = nengo.Network()
    with model: 
        # We want to initialize a list of inputs and a list of actions
        
        motor_network = make_motor_network(input_items, action_items)
        input_list = load_actions(vocab = motor_network.motor_vocab, items = input_items)
        # This box knows what to do for each action (mapping between input list and action list)
        # returns output action rector. We pass that to the finger state.
        #finger state connected to disp_node, which is going to show us stuff
    
        #disp = spa.State(D)
        #nengo.Connection(motor_network.finger_pos, disp)
        
        def present_vector(t):
            if t < 2:
                return input_list[0]
            else:
                return input_list[1]
            
        stim = nengo.Node(present_vector)
        
        nengo.Connection(stim, motor_network.motor_input.input)
        
