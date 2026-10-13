# Import libraries
from keras import Input, Model
from keras.layers import BatchNormalization, Dropout, Dense, Activation
 
from termcolor import cprint

from swamp.helpers import bitify, create_blok

"""
Function creating a deep autoencoder to reduce dimensionality
"""
def create_autoencoder(n_inputs, n_bottleneck, drop_rate, activations, depth =4, optimizer = 'Adam', loss = 'mse', output_activation = 'tanh'):
    if depth != 0:
        # Create array of number of nodes per layer of the encoder and decoder
        neurons_per_layer_e = []
        
        # Bitify because multiples of 8 makes it easier to batch for your cpu
        neurons_per_layer_e.append(int(bitify(n_inputs)))
        depth = depth + 1
        for i in range(depth-1):
            neurons_per_layer_e.append(int(bitify(n_inputs*(depth-(i+1))/(depth))))
        neurons_per_layer_e = neurons_per_layer_e[1:]
        neurons_per_layer_d = neurons_per_layer_e[::-1]
        depth = depth - 1
        # Print the full model
        full_model = [n_inputs] + neurons_per_layer_e + [n_bottleneck] + neurons_per_layer_d + [n_inputs]
        full_model = ' - '.join([str(int) for int in full_model])
        cprint(f"Structure of the Autoencoder model:","blue")
        cprint(f"{full_model}","cyan")
        
        # Inputs of the encoder
        input_e = Input(shape=(n_inputs,), name = 'inputs encoder')
        
        # Create encoder
        e = create_blok(input_e, neurons_per_layer_e[0], drop_rate, activations)
        
        # Go deep
        for i in range(depth-1):
            e = create_blok(e, neurons_per_layer_e[i+1], drop_rate, activations)
        bottle = Dense(n_bottleneck)(e)
        bottle = BatchNormalization()(bottle)
        bottle = Activation(activations)(bottle)
        bottle = BatchNormalization()(bottle)
        encoder = Model(inputs=input_e, outputs = bottle, name = 'encoder')
        
        # Create decoder
        input_d = Input(shape=(n_bottleneck,))
        d = create_blok(input_d, neurons_per_layer_d[0], drop_rate, activations)
        
        # Deeper
        for i in range(depth-1):
            d = create_blok(d, neurons_per_layer_d[i+1], drop_rate, activations)
        output_d = Dense(n_inputs, activation= output_activation)(d)
        decoder = Model(inputs = input_d, outputs = output_d, name = 'decoder')
        
        # Create output of the autoencoder
        output_ae = decoder(encoder(input_e))
        
        # Create the autoencoder
        model = Model(inputs=input_e, outputs=output_ae, name = 'autoencoder')
        model.compile(optimizer, loss, metrics = ["accuracy","mse"])

        return model
    else:
        """
        If you want just the bottleneck
        """
        # Print the full model
        full_model = [n_inputs] + [n_bottleneck] + [n_inputs]
        full_model = ' - '.join([str(int) for int in full_model])
        cprint(f"Structure of the Autoencoder model:","blue")
        cprint(f"{full_model}","cyan")
        
        input_e = Input(shape=(n_inputs,), name = 'inputs encoder')
        bottle = Dense(n_bottleneck)(input_e)
        bottle = Dropout(drop_rate)(bottle)
        bottle = BatchNormalization()(bottle)
        bottle = Activation(activations)(bottle)
        bottle = BatchNormalization()(bottle)
        encoder = Model(inputs=input_e, outputs = bottle, name = 'encoder')

        input_d = Input(shape=(n_bottleneck,))
        output_d = Dense(n_inputs, activation= output_activation)(input_d)
        decoder = Model(inputs = input_d, outputs = output_d, name = 'decoder')
        
        # Create output of the autoencoder
        output_ae = decoder(encoder(input_e))
        
        # Create the autoencoder
        model = Model(inputs=input_e, outputs=output_ae, name = 'autoencoder')
        model.compile(optimizer, loss, metrics = ["accuracy","mse"])
        return model
        
def create_decoder(n_inputs, neurons, n_bottleneck, drop_rate, activations, depth =4, optimizer = 'Adam', loss = 'mse', output_activation = 'tanh'):
    
    # Create array of number of nodes per layer of the encoder and decoder
    neurons_per_layer_e = []
    
    # Bitify because multiples of 8 makes it easier to batch for your cpu
    neurons_per_layer_e.append(int(bitify(n_inputs)))
    depth = depth + 1
    for i in range(depth-1):
        neurons_per_layer_e.append(int(bitify(n_inputs*(depth-(i+1))/(depth))))
    neurons_per_layer_e = neurons_per_layer_e[1:]
    neurons_per_layer_d = neurons_per_layer_e[::-1]
    depth = depth - 1
    # Print the full model
    full_model = [n_bottleneck] + neurons_per_layer_d + [n_inputs]
    full_model = ' - '.join([str(int) for int in full_model])
    cprint(f"Structure of the Autoencoder model:","blue")
    cprint(f"{full_model}","cyan")
    
    # Create decoder
    input_d = Input(shape=(n_bottleneck,))
    d = create_blok(input_d, neurons_per_layer_d[0], drop_rate, activations)
    
    # Deeper
    for i in range(depth-1):
        d = create_blok(d, neurons_per_layer_d[i+1], drop_rate, activations)
    output_d = Dense(n_inputs, activation= output_activation)(d)
    decoder = Model(inputs = input_d, outputs = output_d, name = 'decoder')
    
    # Create output of the autoencoder
    output_decoder = decoder(input_d)
    
    # Create the autoencoder
    model = Model(inputs=input_d, outputs=output_decoder, name = 'autoencoder')
    model.compile(optimizer, loss, metrics = ["accuracy","mse"])
    print(model.summary())
    return model