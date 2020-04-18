import time
import numpy as np

# optimization problem
value = np.array([[12,10,9,5,3,4,11,17,3,2],[12,10,9,5,3,4,11,17,3,2],[12,10,9,5,3,4,11,17,3,2]])
weight = np.array([[1,5,3,4,15,20,13,7,3,4],[4,1,18,8,11,6,9,7,6,5],[14,11,10,2,1,3,2,17,1,8]])


# capacities
A = 17
B = 14
C = 30

# punishment
punish_A = lambda x: min(sum(x[:,0]*weight[0,:]) - PFB, 0) * -9999
punish_B = lambda x: min(sum(x[:,1]*weight[1,:]) - CB, 0) * -9999
punish_C = lambda x: min(sum(x[:,2]*weight[2,:]) - ELA, 0) * -9999

# objective function
f = lambda x: sum(np.diag(np.dot(x,value))) + punish_A(x) + punish_B(x) + punish_C(x)

# settings for method
FE = 60000
popSize = 50
maxGen = int(FE/popSize)
prob_mut = 0.1
prob_xo = 0.5
N = 10

# decision variables

x=np.zeros((popSize,N,3)) #(debt, rows, columns)

for d in range(popSize):
    items = np.random.randint(1,N+1)
    c = np.random.randint(3, size=items)
    r = np.random.choice(N, size=items, replace=False)
    x[d,r,c] = 1

x_cur = x

f_cur = np.zeros(popSize)* np.nan
for d in range(popSize):
    f_cur[d] = f(x_cur[d,:,:])

f_el = min(f_cur)
i_el = np.where(f_cur == f_el)
x_el = x_cur[i_el[0],:,:]


for gen in range(maxGen):

    # cross over
    p2 = np.random.permutation(popSize)
    x_new = np.copy(x_cur)
    XO = np.random.rand(popSize, N,1) < prob_xo
    XO = np.concatenate((XO, XO, XO), axis=2)
    x_cur_perm = x_cur[p2,:,:]
    x_new[XO] = x_cur_perm[XO]

    # mutation
    d,r,c = np.where(np.random.rand(popSize,N,3) < prob_mut)
    x_new[d,r,:] = 0
    x_new[d,r,c] = 1
    for d in range(popSize):
        ind = np.where(np.sum(x_new[d,:,:] ,axis = 1) > 1)
        x_new[d,ind,:] = 0
        x_new[d,ind,np.random.randint(3)] = 1

    # selection
    f_new = np.zeros(popSize)* np.nan
    for d in range(popSize):
        f_new[d] = f(x_new[d,:,:])
    replace = f_new < f_cur
    x_cur[replace, :,:] = x_new[replace,: ,:]
    f_cur[replace] = f_new[replace]

    # elitist
    fe = min(f_cur)
    i_el = np.where(f_cur == f_el)
    if fe < f_el:
        f_el = fe
        x_el = x_cur[i_el[0],: ,:]

    print("generation: ", gen, "elitist: ", f_el,"**", "f_cur: ", f_cur)
    #lowest is 33**

    # time.sleep(0.5)
