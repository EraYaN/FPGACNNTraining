import argparse
import time
import test_nn_framework as tnnf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPGA bitstream loading from python")
    parser.add_argument("kernel_file", type=argparse.FileType('rb'))

    args = parser.parse_args()

    cpu_engine = tnnf.CPUEngine()
    fpga_engine = tnnf.FPGAEngine(args.kernel_file)

    cpu_engine.create_buffers()
    fpga_engine.create_buffers()

    cpu_time = cpu_engine.test()
    fpga_time = fpga_engine.test()

    fpga_engine.finish_device_queue()

    batches = int(cpu_engine.x_test.shape[0] / cpu_engine.testbatch_size)
    ops = cpu_engine.get_testops()
    # print("FPGA Time: {0} usec. {1:.3f} GFLOPS".format(fpga_time / batches * 1e6,
    #                                                    ops / 1e9 / (fpga_time / batches)))
    print("CPU Time: {0} usec. {1:.3f} GFLOPS".format(cpu_time * 1e6,
                                                      ops / 1e9 / cpu_time))

    fpga_engine.verify_results(cpu_engine.get_output())
