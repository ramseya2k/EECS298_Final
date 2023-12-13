import numpy as np
import struct
from time import time
from pynq import Overlay
from pynq import allocate

# Load the LeNet-5 overlay
overlay = Overlay("/path/to/your/lenet_overlay.bit")

# Instantiate the LeNet-5 FPGA accelerator
lenet_fpga_inst = overlay.pl_lenet5_fpga_0

# Max number of test samples in LeNet is 10,000
NUM_TESTS = 10000

# Static allocation of test images
images = np.zeros((NUM_TESTS, 28, 28), dtype=np.uint8)
labels = np.zeros(NUM_TESTS, dtype=np.uint8)

# Static allocation of network parameters and their outputs
image = allocate(shape=(32, 32), dtype=np.float32)
conv1_weights = allocate(shape=(6, 1, 5, 5), dtype=np.float32)
conv1_bias = allocate(shape=(6,), dtype=np.float32)
conv1_output = allocate(shape=(6, 28, 28), dtype=np.float32)
pool2_output = allocate(shape=(6, 14, 14), dtype=np.float32)
conv3_weights = allocate(shape=(16, 6, 5, 5), dtype=np.float32)
conv3_bias = allocate(shape=(16,), dtype=np.float32)
conv3_output = allocate(shape=(16, 10, 10), dtype=np.float32)
pool4_output = allocate(shape=(16, 5, 5), dtype=np.float32)
conv5_weights = allocate(shape=(120, 16, 5, 5), dtype=np.float32)
conv5_bias = allocate(shape=(120,), dtype=np.float32)
conv5_output = allocate(shape=(120, 1, 1), dtype=np.float32)
fc6_weights = allocate(shape=(10, 120), dtype=np.float32)
fc6_bias = allocate(shape=(10,), dtype=np.float32)
fc6_output = allocate(shape=(10,), dtype=np.float32)

# Function definitions
def relu(input):
    return np.maximum(0, input)

def convolution1(input, weights, bias, output):
    for co in range(6):
        for h in range(28):
            for w in range(28):
                conv_sum = np.sum(weights[co, 0, :, :] * input[0, h:h+5, w:w+5])
                output[co, h, w] = conv_sum + bias[co]

def relu1(input, output):
    output[:] = relu(input)

def max_pooling2(input, output):
    for c in range(6):
        for h in range(14):
            for w in range(14):
                output[c, h, w] = np.max(input[c, h*2:h*2+2, w*2:w*2+2])

def relu2(input, output):
    output[:] = relu(input)

def convolution3(input, weights, bias, output):
    for co in range(16):
        for h in range(10):
            for w in range(10):
                conv_sum = np.sum(weights[co, :, :, :] * input[:, h:h+5, w:w+5])
                output[co, h, w] = conv_sum + bias[co]

def relu3(input, output):
    output[:] = relu(input)

def max_pooling4(input, output):
    for c in range(16):
        for h in range(5):
            for w in range(5):
                output[c, h, w] = np.max(input[c, h*2:h*2+2, w*2:w*2+2])

def relu4(input, output):
    output[:] = relu(input)

def convolution5(input, weights, bias, output):
    for co in range(120):
        conv_sum = np.sum(weights[co, :, :, :] * input[:, :, :])
        output[co, 0, 0] = conv_sum + bias[co]

def relu5(input, output):
    output[:] = relu(input)

def fc6(input, weights, bias, output):
    output[:] = np.dot(weights.reshape(10, 120), input.reshape(120)) + bias

def relu6(input, output):
    output[:] = relu(input)

# Function to offload Convolution Layer 3 and Fully Connected Layer 6 to FPGA
def offload_to_fpga(image, conv3_weights, conv3_bias, conv3_output, fc6_weights, fc6_bias, fc6_output):
    # Synchronize input data to the FPGA
    image.sync_to_device()
    conv3_weights.sync_to_device()
    conv3_bias.sync_to_device()
    fc6_weights.sync_to_device()
    fc6_bias.sync_to_device()

    # Configure LeNet-5 FPGA accelerator
    lenet_fpga_inst.write(0x10, image.physical_address)
    lenet_fpga_inst.write(0x1C, conv3_weights.physical_address)
    lenet_fpga_inst.write(0x20, conv3_bias.physical_address)
    lenet_fpga_inst.write(0x28, fc6_weights.physical_address)
    lenet_fpga_inst.write(0x2C, fc6_bias.physical_address)
    lenet_fpga_inst.write(0x30, conv3_output.physical_address)
    lenet_fpga_inst.write(0x38, fc6_output.physical_address)

    # Start LeNet-5 FPGA accelerator
    lenet_fpga_inst.write(0x00, 1)

    # Wait for accelerator to finish
    while lenet_fpga_inst.read(0x00) == 14:
        pass

    # Read accelerator status (optional)
    accelerator_status = lenet_fpga_inst.read(0x00)
    print("Accelerator Status:", accelerator_status)

    # Synchronize output data
    conv3_output.sync_from_device()
    fc6_output.sync_from_device()

# Function to run the remaining layers on the CPU
def cpu_inference(conv5_weights, conv5_bias, conv5_output, fc6_weights, fc6_bias, fc6_output):
    # Continue inference on CPU
    convolution5(pool4_output, conv5_weights, conv5_bias, conv5_output)
    relu5(conv5_output, conv5_output)

    fc6(conv5_output, fc6_weights, fc6_bias, fc6_output)
    relu6(fc6_output, fc6_output)

# Function to parse MNIST images
def parse_mnist_images(filename, images):
    with open(filename, 'rb') as file:
        _ = struct.unpack('>I', file.read(4))  # magic number
        _ = struct.unpack('>I', file.read(4))  # number of images
        _ = struct.unpack('>I', file.read(4))  # number of rows
        _ = struct.unpack('>I', file.read(4))  # number of columns
        images[:] = np.frombuffer(file.read(NUM_TESTS * 28 * 28), dtype=np.uint8).reshape((NUM_TESTS, 28, 28))

# Function to parse MNIST labels
def parse_mnist_labels(filename, labels):
    with open(filename, 'rb') as file:
        _ = struct.unpack('>I', file.read(4))  # magic number
        _ = struct.unpack('>I', file.read(4))  # number of labels
        labels[:] = np.frombuffer(file.read(NUM_TESTS), dtype=np.uint8)

# Function to parse network parameters (modified to use 'rb' mode for reading binary files)
def parse_parameters(filename):
    with open(filename, 'rb') as file:
        conv1_weights[:] = np.fromfile(file, dtype=np.float32, count=150).reshape((6, 1, 5, 5))
        conv1_bias[:] = np.fromfile(file, dtype=np.float32, count=6)
        conv3_weights[:] = np.fromfile(file, dtype=np.float32, count=2400).reshape((16, 6, 5, 5))
        conv3_bias[:] = np.fromfile(file, dtype=np.float32, count=16)
        conv5_weights[:] = np.fromfile(file, dtype=np.float32, count=48000).reshape((120, 16, 5, 5))
        conv5_bias[:] = np.fromfile(file, dtype=np.float32, count=120)
        fc6_weights[:] = np.fromfile(file, dtype=np.float32, count=1200).reshape((10, 120, 1, 1))
        fc6_bias[:] = np.fromfile(file, dtype=np.float32, count=10)

# Function to get test image from dataset
def get_image(images, idx, image):
    image[:] = -1.0
    image[0, 2:30, 2:30] = images[idx] / 255.0 * 2.0 - 1.0

if __name__ == "__main__":
    print("Starting LeNet")

    print("Parsing MNIST images")
    parse_mnist_images("images.bin", images)

    print("Parsing MNIST labels")
    parse_mnist_labels("labels.bin", labels)

    print("Parsing parameters")
    parse_parameters("params.bin")

    print("Running inference")
    num_correct = 0

    # Starting time
    t1 = time()

    for k in range(NUM_TESTS):
        # Get test image from dataset
        get_image(images, k, image)

        # Begin inference here.
        convolution1(image, conv1_weights, conv1_bias, conv1_output)
        relu1(conv1_output, conv1_output)

        max_pooling2(conv1_output, pool2_output)
        relu2(pool2_output, pool2_output)

        offload_to_fpga(pool2_output, conv3_weights, conv3_bias, conv3_output, fc6_weights, fc6_bias, fc6_output)

        # Run the remaining layers on CPU
        max_pooling4(conv3_output, pool4_output)
        relu4(pool4_output, pool4_output)

        cpu_inference(conv5_weights, conv5_bias, conv5_output, fc6_weights, fc6_bias, fc6_output)

        # Index of the largest output is result
        # Check which output was the largest.
        result = np.argmax(fc6_output)

        if result == labels[k]:
            num_correct += 1

    # Ending time
    t2 = time()
    time_span = t2 - t1

    print("\nTotal Execution Time:   {:.6f} seconds ({:d} images)".format(time_span, NUM_TESTS))
    print("Average Time per Image: {:.6f} seconds".format(time_span / NUM_TESTS))
    print("\nAccuracy = {:.2f}%".format(num_correct / NUM_TESTS * 100.0))
