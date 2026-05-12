import numpy as np
import subprocess
import os

def generate_test(n):
    M = 100 * np.random.rand(n, n)
    # Needs to be symmetric positive definite
    A = M.transpose() @ M + n * np.eye(n)
    x = 100 * np.random.rand(n)
    b = A @ x
    return A, x, b

def generate_test_vec(n, num):
    return [100 * np.random.rand(n) for _ in range(num)]
    

def test_cg(num):
    for _ in range(num):
        proc = subprocess.Popen(["./test"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             text=True)
        
        # Get current n from cpp output
        line = proc.stdout.readline()
        print(line)
        cur_n = int(line.split(" ")[-1])
        print("cur n: ", cur_n)

        line = proc.stdout.readline()

        A, x, b = generate_test(cur_n)
        np.savetxt("../data/A.data", A.reshape(-1))
        np.savetxt("../data/x.data", x)

        proc.stdin.write('\n')
        proc.stdin.flush()
        
        lines = []
        while "duration" not in (line := proc.stdout.readline()):
            lines.append(line[:-1])

        k = int(lines[0].split(' ')[-1])
        start_cg = lines.index("cg solution:") + 1
        start_real = lines.index("real solution:") + 1
        cg = np.fromiter(map(float, lines[start_cg : start_cg + cur_n]), dtype=float)
        real = np.fromiter(map(float, lines[start_cg : start_cg + cur_n]), dtype=float)

        print("return value: ", k)
        print("max error: ", np.max(np.abs(cg - real)))
        print("duration: ", line.split(" ")[-1])

def test_add(num):
    for _ in range(num):
        proc = subprocess.Popen(["./test"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             text=True)
        
        # Get current n from cpp output
        line = proc.stdout.readline()
        cur_n = int(line.split(" ")[-1])
        print("cur n: ", cur_n)

        line = proc.stdout.readline()

        u, v = generate_test_vec(cur_n, 2)
        np.savetxt("../data/vec1.data", u)
        np.savetxt("../data/vec2.data", v)

        proc.stdin.write('\n')
        proc.stdin.flush()

        print(proc.stdout.readline())

if __name__ == "__main__":
    test_add(5)
