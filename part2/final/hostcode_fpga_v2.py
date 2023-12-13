import numpy as np
import struct
from time import time
from pynq import Overlay, allocate

# Max number of test samples in LeNet is 10,000
NUM_TESTS = 10000

# Load the overlay with the fc6 accelerator
overlay = Overlay("/path/to/your/fc6_bitstream.bit")

# Access the fc6 IP instance in the overlay
fc6_inst = overlay.fc6_0

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

def fc6_accelerator(input, weights, bias, output):
    input_buf = allocate(input.shape, dtype=np.float32)
    weights_buf = allocate(weights.shape, dtype=np.float32)
    output_buf = allocate(output.shape, dtype=np.float32)

    input_buf[:] = input
    weights_buf[:] = weights

    fc6_inst.write(0x10, input_buf.physical_address)
    fc6_inst.write(0x14, 0)
    fc6_inst.write(0x1C, weights_buf.physical_address)
    fc6_inst.write(0x20, 0)
    fc6_inst.write(0x28, output_buf.physical_address)
    fc6_inst.write(0x2C, 0)

    fc6_inst.write(0x00, 1)  # Start accelerator
    while fc6_inst.read(0x00) == 14:  # Wait for completion
        pass

    output[:] = output_buf

    input_buf.freebuffer()
    weights_buf.freebuffer()
    output_buf.freebuffer()

def relu6(input, output):
    output[:] = relu(input)

def parse_mnist_images(filename, images):
    with open(filename, 'rb') as file:
        _ = struct.unpack('>I', file.read(4))  # magic number
        _ = struct.unpack('>I', file.read(4))  # number of images
        _ = struct.unpack('>I', file.read(4))  # number of rows
        _ = struct.unpack('>I', file.read(4))  # number of columns
        images[:] = np.frombuffer(file.read(NUM_TESTS * 28 * 28), dtype=np.uint8).reshape((NUM_TESTS, 28, 28))

def parse_mnist_labels(filename, labels):
    with open(filename, 'rb') as file:
        _ = struct.unpack('>I', file.read(4))  # magic number
        _ = struct.unpack('>I', file.read(4))  # number of labels
        labels[:] = np.frombuffer(file.read(NUM_TESTS), dtype=np.uint8)

def parse_parameters(filename):
    with open(filename, 'rb') as file:
        conv1_weights[:] = np.frombuffer(file.read(150 * 4), dtype=np.float32).reshape((6, 1, 5, 5))
        conv1_bias[:] = np.frombuffer(file.read(6 * 4), dtype=np.float32)
        conv3_weights[:] = np.frombuffer(file.read(2400 * 4), dtype=np.float32).reshape((16, 6, 5, 5))
        conv3_bias[:] = np.frombuffer(file.read(16 * 4), dtype=np.float32)
        conv5_weights[:] = np.frombuffer(file.read(48000 * 4), dtype=np.float32).reshape((120, 16, 5, 5))
        conv5_bias[:] = np.frombuffer(file.read(120 * 4), dtype=np.float32)
        fc6_weights[:] = np.frombuffer(file.read(1200 * 4), dtype=np.float32).reshape((10, 120, 1, 1))
        fc6_bias[:] = np.frombuffer(file.read(10 * 4), dtype=np.float32)

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

    # starting time
    t1 = time()

    for k in range(NUM_TESTS):
        # Get test image from dataset
        get_image(images, k, image)

        # Begin inference here.
        convolution1(image, conv1_weights, conv1_bias, conv1_output)
        relu1(conv1_output, conv1_output)

        max_pooling2(conv1_output, pool2_output)
        relu2(pool2_output, pool2_output)

        convolution3(pool2_output, conv3_weights, conv3_bias, conv3_output)
        relu3(conv3_output, conv3_output)

        max_pooling4(conv3_output, pool4_output)
        relu4(pool4_output, pool4_output)

        convolution5(pool4_output, conv5_weights, conv5_bias, conv5_output)
        relu5(conv5_output, conv5_output)

        fc6_input = conv5_output.reshape((120, 1, 1)).astype(np.float32)
        fc6_output[:] = 0.0

        fc6_accelerator(fc6_input, fc6_weights, fc6_bias, fc6_output)

        relu6(fc6_output, fc6_output)

        # Print or use the result as needed
        # ... (you can print or process fc6_output)

        # Index of the largest output is result
        # Check which output was the largest.
        result = np.argmax(fc6_output)

        if result == labels[k]:
            num_correct += 1

    # ending time
    t2 = time()
    time_span = t2 - t1

    print("\nTotal Execution Time:   {:.6f} seconds ({:d} images)".format(time_span, NUM_TESTS))
    print("Average Time per Image: {:.6f} seconds".format(time_span / NUM_TESTS))
    print("\nAccuracy = {:.2f}%".format(num_correct / NUM_TESTS * 100.0))
