import theseus as th
import torch
import torch.nn as nn
import time
seed = 0
torch.random.manual_seed(seed)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # some arbitrary NN
        self.nn = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))

        # Add a theseus layer with a single cost function whose error depends on the NN
        objective = th.Objective()
        x = th.Vector(2, name="x")
        y = th.Vector(1, name="y")
        objective.add(th.AutoDiffCostFunction([x], self._error_fn, 1, aux_vars=[y]))
        optimizer = th.LevenbergMarquardt(objective, th.CholeskyDenseSolver, max_iterations=50,step_size=0.0001) 
        #LUDenseSolver LUCudaSparseSolver CholeskyDenseSolver CholmodSparseSolver BaspachoSparseSolver
        #optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver, max_iterations=100,step_size=0.001) #LUDenseSolver CholeskyDenseSolver
        #optimizer = th.Dogleg(objective, th.CholeskyDenseSolver, max_iterations=50,step_size=0.001) #LUDenseSolver CholeskyDenseSolver
        self.layer = th.TheseusLayer(optimizer)
        self.layer.to(device='cuda', dtype=torch.float32)

    def _error_fn(self, optim_vars, aux_vars):
        x = optim_vars[0].tensor
        y = aux_vars[0].tensor
        x = torch.clamp(x, -1, 1,) 
        err = - self.nn(x) + 1000 #+ penalty
        return err

    # Run theseus so that NN(x*) is close to y
    def forward(self, y):
        x0 = torch.zeros(y.shape[0], 2,device='cuda')

        sol, info = self.layer.forward(
            {"x": x0, "y": y}, optimizer_kwargs={"track_best_solution": True, 
				"verbose": True,"damping": 1.0, "backward_mode": "truncated", "backward_num_iterations": 5,}
        )
        print("sol:",sol)
        print("info:",info)
        return sol["x"]



m = Model().to(device='cuda', dtype=torch.float32)
optim = torch.optim.Adam(m.nn.parameters(), lr=0.01)
y = torch.ones(1, 1, device='cuda')
#xopt = m.forward(y)
#print("xopt:", xopt)

start = time.time()
for i in range(5):
    optim.zero_grad()
    xopt = m.forward(y)
    print("x1:",xopt)
    loss = (xopt**2).sum()*0.1
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(m.nn.parameters(), 1000, error_if_nonfinite=False)
    print("grad_norm:",grad_norm)
    # import pdb
    # pdb.set_trace()
    optim.step()
    print("x2:",xopt)
    print("Outer loss:", loss.item(), "\n------------------------")
print("total time:", time.time() - start)
