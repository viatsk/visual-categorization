import nengo
import os
import numpy as np
import png
import itertools
from nengo import spa
from nengo_extras.vision import Gabor, Mask

import base64
import PIL.Image
import cStringIO



def load_images(dir, items):
    files = os.listdir(dir)
    files2 = []
    for fn in files:
        if fn[-4:] == '.png' and (fn[:-4] in items):
            files2.append(fn)

    X_train = np.empty(shape=(np.size(files2), 90*14),dtype='float32')
    for i,fn in enumerate(files2):
            r = png.Reader(dir + fn)
            r = r.asDirect()
            image_2d = np.vstack(itertools.imap(np.uint8, r[2]))
            image_2d /= 255
            image_1d = image_2d.reshape(1,90*14)
            X_train[i] = image_1d
    
    X_train = 2 * X_train - 1  # normalize to -1 to 1
    return X_train

def vector_gen_function(items, vocab):
    output_list = []
    for i in items:
        c = vocab.parse(i).v
        output_list.append(c)
    return np.array(output_list)

def display_func(t, x):

    if np.size(x) > 14*90:
        input_shape = (1, 28, 90)
    else:
        input_shape = (1,14,90)

    values = x.reshape(input_shape)
    values = values.transpose((1, 2, 0))
    values = (values + 1) / 2 * 255.
    values = values.astype('uint8')

    if values.shape[-1] == 1:
        values = values[:, :, 0]

    png = PIL.Image.fromarray(values)
    buffer = cStringIO.StringIO()
    png.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())

    display_func._nengo_html_ = '''
           <svg width="100%%" height="100%%" viewbox="0 0 %i %i">
           <image width="100%%" height="100%%"
                  xlink:href="data:image/png;base64,%s"
                  style="image-rendering: auto;">
           </svg>''' % (input_shape[2]*2, input_shape[1]*2, ''.join(img_str))


def make_vision_system(images, outputs, n_neurons = 1000, AIT_V1_strength = 0.06848695023305285, V1_r_transform = 0.11090645719111913, AIT_r_transform = 0.8079719992231219):

     #represent currently attended item
    vision_system = nengo.Network(label = 'vision_system')
    with vision_system:
        presentation_node = nengo.Node(None, size_in = images.shape[1], label = 'presentation_node')
        vision_system.presentation_node = presentation_node
        rng = np.random.RandomState(9)
        encoders = Gabor().generate(n_neurons, (11, 11), rng=rng)  # gabor encoders, work better, 11,11 apparently, why?
        encoders = Mask((14, 90)).populate(encoders, rng=rng, flatten=True)
        
        V1 = nengo.Ensemble(n_neurons, images.shape[1], eval_points=images,
                                                neuron_type=nengo.LIFRate(),
                                                intercepts=nengo.dists.Choice([-0.5]), #can switch these off
                                                max_rates=nengo.dists.Choice([100]),  # why?
                                                encoders=encoders,
                                                label = 'V1')
                                                                    #  1000 neurons, nrofpix = dimensions
        # visual_representation = nengo.Node(size_in=Dmid) #output, in this case 466 outputs
        AIT = nengo.Ensemble(n_neurons, dimensions=outputs.shape[1], label = 'AIT')  # output, in this case 466 outputs
        
        visconn = nengo.Connection(V1, AIT, synapse=0.005,
                                        eval_points = images, function=outputs,
                                        solver=nengo.solvers.LstsqL2(reg=0.01))
        Ait_V1_backwardsconn = nengo.Connection(AIT,V1, synapse = 0.005, 
                                        eval_points = outputs, function = images,
                                        solver=nengo.solvers.LstsqL2(reg=0.01), transform = AIT_V1_strength) #Transform makes this connection a lot weaker then the forwards conneciton
        nengo.Connection(presentation_node, V1, synapse=None)
        nengo.Connection(AIT, AIT, synapse = 0.1, transform = AIT_r_transform)
        nengo.Connection(V1, V1, synapse = 0.1, transform = V1_r_transform)
        
        # display attended item
        display_node = nengo.Node(display_func, size_in=presentation_node.size_out, label = 'display_node')  # to show input
        nengo.Connection(presentation_node, display_node, synapse=None)
        
        # THESE PIECES MAKE EVERYTHING WORK please dont touch them
        vision_system.AIT = AIT
        vision_system.V1 = V1
        
    return vision_system

if __name__ == "__builtin__":
    D = 32
    items = ['WHISKEY','FATIGUE']
    vocab = spa.Vocabulary(D)
    model = nengo.Network()
    with model:
        directory = '/home/stacy/github/visual-categorization/assoc_recog_s/images/'
        image_list = load_images(directory, items = items)
        print(image_list.shape)
        output_list = vector_gen_function(items, vocab = vocab)
        #print(output_list)
        vision_system = make_vision_system(image_list, output_list) # the output list is the spa vector list corresponding to the inputed images
        
        def present_func(t):
            index = int(t/0.1)
            return image_list[index % len(image_list)]
            
        stim = nengo.Node(present_func)
        nengo.Connection(stim, vision_system.presentation_node)
        test = spa.State(D, vocab = vocab)
        
        nengo.Connection(vision_system.AIT, test.input)
