import numpy as np
import struct
from time import time
from pynq import Overlay
from pynq import allocate

# Load the overlay containing your FPGA accelerator
overlay = Overlay("your_overlay.bit")
overlay.download()

# Max number of test samples in LeNet is 10,000
NUM_TESTS = 10000

# Static allocation of test images
images = np.zeros((NUM_TESTS, 28, 28), dtype=np.uint8)
labels = np.zeros(NUM_TESTS, dtype=np.uint8)

# Static allocation of network parameters and their outputs
image = np.zeros((1, 32, 32), dtype=np.float32)
conv1_weights = np.zeros((6, 1, 5, 5), dtype=np.float32)
conv1_bias = np.zeros(6, dtype=np.float32)
conv1_output = np.zeros((6, 28, 28), dtype=np.float32)

pool2_output = np.zeros((6, 14, 14), dtype=np.float32)

conv3_weights = np.zeros((16, 6, 5, 5), dtype=np.float32)
conv3_bias = np.zeros(16, dtype=np.float32)
conv3_output = np.zeros((16, 10, 10), dtype=np.float32)

pool4_output = np.zeros((16, 5, 5), dtype=np.float32)

conv5_weights = np.zeros((120, 16, 5, 5), dtype=np.float32)
conv5_bias = np.zeros(120, dtype=np.float32)
conv5_output = np.zeros((120, 1, 1), dtype=np.float32)

fc6_weights = np.zeros((10, 120, 1, 1), dtype=np.float32)
fc6_bias = np.zeros(10, dtype=np.float32)
fc6_output = np.zeros(10, dtype=np.float32)

# Function definitions (previous code)

# Function to call FPGA fc6
def fpga_fc6(input, weights, bias, output, enable):
    if enable:
        fc6_accel = overlay.fc6_accel  # Replace 'fc6_accel' with the actual name in your overlay
        fc6_accel.write(0x10, input.physical_address)
        fc6_accel.write(0x18, weights.physical_address)
        fc6_accel.write(0x20, bias.physical_address)
        fc6_accel.write(0x28, output.physical_address)
        fc6_accel.write(0x00, 0x01)  # Start the accelerator

        # Wait for the accelerator to finish
        while not fc6_accel.read(0x00) & 0x2:
            pass
    else:
        # Run fc6 on CPU
        fc6(input, weights, bias, output)

# Wrapper function combining both CPU and FPGA processing
def combined_inference(images, labels, conv1_weights, conv1_bias, conv3_weights, conv3_bias,
                        conv5_weights, conv5_bias, fc6_weights, fc6_bias):
    print("Starting LeNet")

    print("Parsing MNIST images")
    parse_mnist_images("images.bin", images)

    print("Parsing MNIST labels")
    parse_mnist_labels("labels.bin", labels)

    print("Parsing parameters")
    parse_parameters("params.bin")

    print("Running combined inference")
    num_correct = 0

    # Allocate memory for FPGA acceleration
    fc6_input_buffer = allocate(shape=(120,), dtype=np.float32)
    fc6_weights_buffer = allocate(shape=(10, 120), dtype=np.float32)
    fc6_bias_buffer = allocate(shape=(10,), dtype=np.float32)
    fc6_output_buffer = allocate(shape=(10,), dtype=np.float32)

    # starting time
    t1 = time()

    for k in range(NUM_TESTS):
        # Get test image from dataset
        get_image(images, k, image)

        # Begin inference on CPU
        convolution1(image, conv1_weights, conv1_bias, conv1_output)
        relu1(conv1_output, conv1_output)

        max_pooling2(conv1_output, pool2_output)
        relu2(pool2_output, pool2_output)

        # Call FPGA convolution3
        convolution3(pool2_output, conv3_weights, conv3_bias, conv3_output)
        relu3(conv3_output, conv3_output)

        max_pooling4(conv3_output, pool4_output)
        relu4(pool4_output, pool4_output)

        convolution5(pool4_output, conv5_weights, conv5_bias, conv5_output)
        relu5(conv5_output, conv5_output)

        fc6(conv5_output, fc6_weights, fc6_bias, fc6_output)

        # Call FPGA fc6
        fpga_fc6(fc6_input_buffer, fc6_weights_buffer, fc6_bias_buffer, fc6_output_buffer, enable_fc6)
        # Inference ends here.

        # Index of the largest output is result
        # Check which output was the largest.
        result = np.argmax(fc6_output_buffer)

        if result == labels[k]:
            num_correct += 1

    # Free FPGA memory buffers
    fc6_input_buffer.freebuffer()
    fc6_weights_buffer.freebuffer()
    fc6_bias_buffer.freebuffer()
    fc6_output_buffer.freebuffer()

    # ending time
    t2 = time()
    time_span = t2 - t1

    print("\nTotal Execution Time:   {:.6f} seconds ({:d} images)".format(time_span, NUM_TESTS))
    print("Average Time per Image: {:.6f} seconds".format(time_span / NUM_TESTS))
   
