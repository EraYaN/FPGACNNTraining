import sys

minibatch_size = 4
rows_in = 4
rows_out = 4

N = minibatch_size
M = rows_in
K = rows_out

BLOCK_SIZE = 2

def render_matrix(y,x,y_size,x_size):
    """

    :param y: Row
    :param x: Col
    :param y_size: Number of Rows
    :param x_size: Number of Cols
    :return:
    """
    for j in range(0,y_size):
        sys.stdout.write('|')
        for i in range(0, x_size):
            if i == x and j == y:
                sys.stdout.write(" @ ")
            else:
                sys.stdout.write(" * ")
        sys.stdout.write('|\n')
    sys.stdout.write('\n')


for i0 in range(0,N,BLOCK_SIZE):
    imax = N if i0 + BLOCK_SIZE > N else i0 + BLOCK_SIZE

    for j0 in range(0,M,BLOCK_SIZE):
        jmax = M if j0 + BLOCK_SIZE > M else j0 + BLOCK_SIZE

        for k0 in range(0,K,BLOCK_SIZE):
            kmax = K if k0 + BLOCK_SIZE > K else k0 + BLOCK_SIZE
            print("A: {2:.0f},{0:.0f}; W: {2:.0f},{1:.0f}; O: {1:.0f},{0:.0f}".format(i0/BLOCK_SIZE,j0/BLOCK_SIZE,k0/BLOCK_SIZE))

            render_matrix(i0/BLOCK_SIZE,k0/BLOCK_SIZE,2,2)

            ## Load block ram here. Or NDRange
            print("Load act {1:.0f},{0:.0f}".format(i0 / BLOCK_SIZE, k0/BLOCK_SIZE))
            print("Load weights {1:.0f},{0:.0f}".format(j0 / BLOCK_SIZE, k0 / BLOCK_SIZE))

            for i1 in range(i0,imax):
                mi = M * i1;
                ki = K * i1;

                for j1 in range(j0, jmax):
                    kij = ki + j1;
                    sj = M * j1

                    for k1 in range(k0,kmax):
                        #C[kij] += A[mi + k1] * B[sj + k1];
                        print("A: {0:.0f}; W: {1:.0f}; O: {2:.0f}".format(mi+k1,sj+k1,kij))

        print("Write back output {1:.0f},{0:.0f}".format(i0/BLOCK_SIZE,j0/BLOCK_SIZE))