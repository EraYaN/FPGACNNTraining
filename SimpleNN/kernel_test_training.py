import argparse
import time
import test_nn_framework as tnnf
import numpy as np
import keras


RUN_ON_DEVICE = True
VERIFY = False

def display_verify_result(res, name="Data"):
    total = len(res)
    correct = 0
    for i in res:
        if res[i]:
            correct += 1
    print("{} Verify Result: {} out of {} correct.".format(name, correct, total), res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPGA bitstream loading from python")
    parser.add_argument("kernel_file", type=argparse.FileType('rb'))

    args = parser.parse_args()

    cpu_engine = tnnf.CPUEngine(set_verify_config=VERIFY)
    cpu_engine.create_buffers(pretrained=True)

    if RUN_ON_DEVICE:
        fpga_engine = tnnf.FPGAEngine(args.kernel_file, set_verify_config=VERIFY)
        fpga_engine.create_buffers(pretrained=True)

    print("Running CPU code...")
    cpu_time = cpu_engine.train()
    cpu_engine.set_train_input(0)
    cpu_engine.fw_function()
    

    if RUN_ON_DEVICE:
        print("Running FPGA code...")
        fpga_time = fpga_engine.train()
        fpga_engine.set_train_input(0)
        fpga_engine.fw_function()
        print("Getting all device buffers...")

        fpga_engine.retrieve_buffers_from_device(all=True)

        fpga_engine.finish_device_queue()

        ops = fpga_engine.get_trainops()
        print("FPGA Time: {0} usec. {1:.3f} GFLOPS".format(fpga_time / fpga_engine.epochs / fpga_engine.minibatch_size * 1e6, ops / 1e9 / (fpga_time / fpga_engine.epochs / fpga_engine.minibatch_size)))
        print("CPU Time: {0} usec. {1:.3f} GFLOPS".format(cpu_time / cpu_engine.epochs / cpu_engine.minibatch_size * 1e6, ops / 1e9 / (cpu_time / cpu_engine.epochs / cpu_engine.minibatch_size)))

        # print("Verifying...")

        # dW_matches = {}
        # bias_matches = {}
        # weights_matches = {}
        # delta_matches = {}
        # act_matches = {}
        #
        # for layer in range(0, fpga_engine.layers):
        #     print("Verifying dW[{}]".format(layer))
        #     dW_matches[layer] = fpga_engine.verify_dW(cpu_engine, layer=layer)
        #     print("Verifying bias[{}]".format(layer))
        #     bias_matches[layer] = fpga_engine.verify_bias(cpu_engine, layer=layer)
        #     print("Verifying weights[{}]".format(layer))
        #     weights_matches[layer] = fpga_engine.verify_weights(cpu_engine, layer=layer)
        #     if layer > 0:
        #         print("Verifying delta[{}]".format(layer))
        #         delta_matches[layer] = fpga_engine.verify_delta(cpu_engine, layer=layer)
        #     print("Verifying act[{}]".format(layer))
        #     act_matches[layer] = fpga_engine.verify_act(cpu_engine, layer=layer)
        #
        # print("Verifying delta[{}]".format(fpga_engine.layers))
        # delta_matches[fpga_engine.layers] = fpga_engine.verify_delta(cpu_engine, layer=fpga_engine.layers)
        # print("Verifying act[{}]".format(fpga_engine.layers))
        # act_matches[fpga_engine.layers] = fpga_engine.verify_act(cpu_engine, layer=fpga_engine.layers)
        #
        # display_verify_result(dW_matches, "dW")
        # display_verify_result(bias_matches, "bias")
        # display_verify_result(weights_matches, "weights")
        # display_verify_result(delta_matches, "delta")
        # display_verify_result(act_matches, "act")
