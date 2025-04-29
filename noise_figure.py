import numpy as np
from mindquantum.core.gates import X, Y, Z, S, H, RX, RY, RZ, Measure, UnivMathGate, DepolarizingChannel
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
import scipy.linalg as la
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import os

def SpinChain(n,J,h):
    X = np.array([[0, 1], [1, 0]],dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]],dtype=complex)
    Z = np.array([[1, 0], [0, -1]],dtype=complex)
    def hopping(P,i):
        assert i < n, "i should be less than n"
        if i == 0 or i == n-1:
            matrix = P
        else:
            matrix = np.eye(2,dtype=complex)
        for j in range(1,n):
            if j == i or j == i+1:
                matrix = np.kron(P,matrix)
            else:
                matrix = np.kron(np.eye(2,dtype=complex),matrix)
        return matrix
    def potential(P,i):
        assert i < n, "i should be less than n"
        if i == 0:
            matrix = P
        else:
            matrix = np.eye(2,dtype=complex)
        for j in range(1,n):
            if j == i:
                matrix = np.kron(P,matrix)
            else:
                matrix = np.kron(np.eye(2,dtype=complex),matrix)
        return matrix
    
    # hopping term
    HoppingX = np.zeros((2**n,2**n),dtype=complex)
    HoppingY = np.zeros((2**n,2**n),dtype=complex)
    HoppingZ = np.zeros((2**n,2**n),dtype=complex)
    if n == 2:
        for i in range(n-1):
            HoppingX += hopping(X,i)*J[0]
            HoppingY += hopping(Y,i)*J[1]
            HoppingZ += hopping(Z,i)*J[2]
    else:
        for i in range(n):
            HoppingX += hopping(X,i)*J[0]
            HoppingY += hopping(Y,i)*J[1]
            HoppingZ += hopping(Z,i)*J[2]
    # potential term
    PotentialX = np.zeros((2**n,2**n),dtype=complex)
    PotentialY = np.zeros((2**n,2**n),dtype=complex)
    PotentialZ = np.zeros((2**n,2**n),dtype=complex)
    for i in range(n):
        PotentialX += potential(X,i)*h[0]
        PotentialY += potential(Y,i)*h[1]
        PotentialZ += potential(Z,i)*h[2]
    return HoppingX+HoppingY+HoppingZ+PotentialX+PotentialY+PotentialZ

def TimeEvolution(H,t):
    eigenv,U = np.linalg.eig(H)
    diag = np.diag(np.exp(-1.j*t*eigenv))
    return U@diag@np.linalg.inv(U)

def Magnetism(n):
    Z = np.array([[1, 0], [0, -1]],dtype=complex)
    def potential(P,i):
        assert i < n, "i should be less than n"
        if i == 0:
            matrix = P
        else:
            matrix = np.eye(2,dtype=complex)
        for j in range(1,n):
            if j == i:
                matrix = np.kron(P,matrix)
            else:
                matrix = np.kron(np.eye(2,dtype=complex),matrix)
        return matrix
    MagnetismZ_array = []
    for i in range(n):
        MagnetismZ_array.append(potential(Z,i))
    return MagnetismZ_array

def SpinCorr(a,b,n):
    Z = np.array([[1, 0], [0, -1]],dtype=complex)
    if a == 0 or b == 0:
        matrix = Z
    else:
        matrix = np.eye(2,dtype=complex)
    for j in range(1,n):
        if j == a or j == b:
            matrix = np.kron(Z,matrix)
        else:
            matrix = np.kron(np.eye(2,dtype=complex),matrix)
    return matrix

def Diag_averg(matrix_set):
    L = matrix_set.shape[0]
    K = matrix_set.shape[1]
    N = L+K-1
    Y = matrix_set
    def sum(seq):
        a = seq[0]
        for i in range(1,len(seq)):
            a += seq[i]
        return a     
    recovered = []
    for i in range(N):
        if i<=L-1:
            seq = [Y[m,i-m]/(i+1) for m in range(i+1)]
            recovered.append(sum(seq))
        elif i>L-1 and i<K-1:
            seq = [Y[m,i-m]/L for m in range(L)]
            recovered.append(sum(seq))
        else:
            seq = [Y[m,i-m]/(N-i) for m in range(i-K+1,L)]
            recovered.append(sum(seq))
    return np.array(recovered)

def TimeSeqRecover(time_sequence_error,order):
    N = len(time_sequence_error)
    L = N//2
    K = N-L+1
    X = []
    for i in range(K):
        X.append(time_sequence_error[i:i+L])
    X = np.array(X).T
    U,Lambda,V = np.linalg.svd(X)
    r = order
    Y = U[:,:r]@np.diag(Lambda[:r])@V[:r,:]
    # print(Lambda[:2*r])
    time_sequence_recover = Diag_averg(Y)
    return time_sequence_recover

if __name__ == '__main__':
    n = 8
    size = 1000
    J = [-1.0,-1.0,-1.5]
    h = [1.5,0.0,0.5]
    Hamiltonian = SpinChain(n,J,h)

    sigma = 4.0
    points = 120
    width = 4*sigma

    time_stamp = np.linspace(-width,width,points+1)
    coeffs = [(width*2/points)*np.exp(-time_stamp[j]**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)) for j in range(points+1)]
    U0_thy = TimeEvolution(Hamiltonian,-width)
    dU = TimeEvolution(Hamiltonian,2*width/points)
    Uset_thy = []
    for i in tqdm(range(points+1)):
        Uset_thy.append(U0_thy)
        U0_thy = dU@U0_thy
    Uset_thy = np.array(Uset_thy)

    OB = SpinCorr(0,4,n)

    initial_state = np.zeros(2**n)
    initial_state[2**n-1] = 1.0
    size = 1000
    target_bit = [i+1 for i in range(n)]

    sim = Simulator('mqvector',n+1)

    error_rate = [0,1e-4,1e-3,1e-2,1e-1,0.12,0.15,0.18,0.2,0.22,0.25,0.28,0.3,0.32]

    for s in range(1,11):
        path = f"./sample{s}"
        if not os.path.exists(path):
            os.mkdir(path)

        for error in tqdm(error_rate,desc='error',colour='blue'):
            p = np.sqrt(error*2/points)
            U0 = TimeEvolution(Hamiltonian,-width)
            dU = TimeEvolution(Hamiltonian,2*width/points)
            Uset = []
            for i in tqdm(range(points+1)):
                randmat = np.random.randn(2**n,2**n)+1.j*np.random.randn(2**n,2**n)
                randmat = randmat*p
                Uset.append(U0)
                U0 = (dU+randmat*p)@U0
            Uset = np.array(Uset)

            # OB = I
            time_serial = []
            for i in tqdm(range(points+1),desc='I',colour='yellow'): # t
                realcirc = Circuit()
                realcirc += H.on(0)
                evol_gate = UnivMathGate('Ut',Uset[i])
                realcirc += evol_gate.on(target_bit,0)
                realcirc += H.on(0)
                realcirc += Measure('q0').on(0)
                
                sim.reset()
                sim.set_qs(np.kron(initial_state,[1.0,0.0]))
                result = sim.sampling(circuit=realcirc,shots=size)
                samples = result.data
                try:
                    zero = samples['0']
                except:
                    zero = 0
                try:
                    one = samples['1']
                except:
                    one = 0
                real_part = (zero-one)/size

                imagcirc = Circuit()
                imagcirc += H.on(0)
                evol_gate = UnivMathGate('Ut',Uset[i])
                imagcirc += evol_gate.on(target_bit,0)
                imagcirc += S.on(0)
                imagcirc += Z.on(0)
                imagcirc += H.on(0)
                imagcirc += Measure('q0').on(0)

                sim.reset()
                sim.set_qs(np.kron(initial_state,[1.0,0.0]))
                result = sim.sampling(circuit=imagcirc,shots=size)
                samples = result.data
                try:
                    zero = samples['0']
                except:
                    zero = 0
                try:
                    one = samples['1']
                except:
                    one = 0
                imag_part = (zero-one)/size

                time_serial.append(real_part+1.j*imag_part)

                del realcirc
                del imagcirc  
            time_serial = np.array(time_serial)  

            path = f"./sample{s}/error{error}"
            if not os.path.exists(path):
                os.mkdir(path)
            np.save(f'{path}/time_serial.npy',time_serial)

            # OB = SpinCorr
            QAF_serial = []
            for i in tqdm(range(points+1),desc='t',colour='green'): # t
                QAF = 0
                for j in tqdm(range(points+1),desc='eta',colour='red'): # eta
                    realcirc = Circuit()
                    realcirc += H.on(0)
                    evol_gate = UnivMathGate('Ut',Uset[i])
                    realcirc += evol_gate.on(target_bit,0)
                    corr_gate = UnivMathGate('Ueta',Uset[j])
                    realcirc += corr_gate.on(target_bit)
                    observ_gate = UnivMathGate('obs',OB)
                    realcirc += observ_gate.on(target_bit,0)
                    realcirc += H.on(0)
                    realcirc += Measure('q0').on(0)
                    
                    sim.reset()
                    sim.set_qs(np.kron(initial_state,[1.0,0.0]))
                    result = sim.sampling(circuit=realcirc,shots=size)
                    samples = result.data
                    try:
                        zero = samples['0']
                    except:
                        zero = 0
                    try:
                        one = samples['1']
                    except:
                        one = 0
                    real_part = (zero-one)/size
                    
                    imagcirc = Circuit()
                    imagcirc += H.on(0)
                    evol_gate = UnivMathGate('Ut',Uset[i])
                    imagcirc += evol_gate.on(target_bit,0)
                    corr_gate = UnivMathGate('Ueta',Uset[j])
                    imagcirc += corr_gate.on(target_bit)
                    observ_gate = UnivMathGate('obs',OB)
                    imagcirc += observ_gate.on(target_bit,0)
                    imagcirc += S.on(0)
                    imagcirc += Z.on(0)
                    imagcirc += H.on(0)
                    imagcirc += Measure('q0').on(0)
                    sim.reset()
                    sim.set_qs(np.kron(initial_state,[1.0,0.0]))
                    result = sim.sampling(circuit=imagcirc,shots=size)
                    samples = result.data
                    try:
                        zero = samples['0']
                    except:
                        zero = 0
                    try:
                        one = samples['1']
                    except:
                        one = 0
                    imag_part = (zero-one)/size
                    QAF += (real_part+1.j*imag_part)*coeffs[j]
                    del realcirc
                    del imagcirc  
                QAF_serial.append(QAF)
            QAF_serial = np.array(QAF_serial)

            path = f"./sample{s}/error{error}"
            if not os.path.exists(path):
                os.mkdir(path)
            np.save(f'{path}/QAF_serial.npy',QAF_serial)
        



