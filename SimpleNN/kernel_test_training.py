import argparse
import time
import test_nn_framework as tnnf
import numpy as np
import keras

def display_verify_result(res,name="Data"):
    total = len(res)
    correct = 0
    for i in res:
        if res[i]:
            correct += 1
    print("{} Verify Result: {} out of {} correct.".format(name, correct, total),res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPGA bitstream loading from python")
    parser.add_argument("kernel_file", type=argparse.FileType('rb'))

    args = parser.parse_args()

    cpu_engine = tnnf.CPUEngine(set_verify_config=True)

    fpga_engine = tnnf.FPGAEngine(args.kernel_file, set_verify_config=True)

    cpu_engine.create_buffers(pretrained=False)

    fpga_engine.create_buffers(pretrained=False)

    np.random.seed(0)
    input_cpu = np.random.randint(1, 4, size=(cpu_engine.minibatch_size,cpu_engine.layer_height[0])).astype(dtype=np.float32)
    ground_truth_cpu = np.random.randint(0, cpu_engine.layer_height[cpu_engine.layers], size=cpu_engine.minibatch_size)
    ground_truth_cpu = keras.utils.to_categorical(ground_truth_cpu, cpu_engine.layer_height[cpu_engine.layers]).astype(dtype=np.float32)

    cpu_engine.set_input(input_cpu, ground_truth_cpu)

    fpga_engine.set_input(np.copy(input_cpu), np.copy(ground_truth_cpu))

    print("Running CPU code...")
    cpu_engine.fw_function()
    cpu_engine.bw_function()

    print("Running FPGA code...")
    fpga_engine.fw_function()
    fpga_engine.bw_function()


    print("Getting all device buffers...")

    fpga_engine.retrieve_buffers_from_device(all=True)

    fpga_engine.finish_device_queue()

    print("Verifying...")

    dW_matches = {}
    bias_matches = {}
    weights_matches = {}
    delta_matches = {}
    act_matches = {}

    for layer in range(0, fpga_engine.layers):
        print("Verifying dW[{}]".format(layer))
        dW_matches[layer] = fpga_engine.verify_dW(cpu_engine,layer=layer)
        print("Verifying bias[{}]".format(layer))
        bias_matches[layer] = fpga_engine.verify_bias(cpu_engine, layer=layer)
        print("Verifying weights[{}]".format(layer))
        weights_matches[layer] = fpga_engine.verify_weights(cpu_engine, layer=layer)
        print("Verifying delta[{}]".format(layer))
        delta_matches[layer] = fpga_engine.verify_delta(cpu_engine, layer=layer)
        print("Verifying act[{}]".format(layer))
        act_matches[layer] = fpga_engine.verify_act(cpu_engine, layer=layer)

    print("Verifying delta[{}]".format(fpga_engine.layers))
    delta_matches[fpga_engine.layers] = fpga_engine.verify_delta(cpu_engine, layer=fpga_engine.layers)
    print("Verifying act[{}]".format(fpga_engine.layers))
    act_matches[fpga_engine.layers] = fpga_engine.verify_act(cpu_engine, layer=fpga_engine.layers)

    display_verify_result(dW_matches,"dW")
    display_verify_result(weights_matches, "weights")
    display_verify_result(delta_matches, "delta")
    display_verify_result(bias_matches, "bias")
    display_verify_result(act_matches, "act")

